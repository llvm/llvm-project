/* Test EDNS handling in the stub resolver.
   Copyright (C) 2016-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#include <errno.h>
#include <netdb.h>
#include <resolv.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <support/check.h>
#include <support/resolv_test.h>
#include <support/support.h>
#include <support/test-driver.h>
#include <support/xthread.h>

/* Data produced by a test query.  */
struct response_data
{
  char *qname;
  uint16_t qtype;
  struct resolv_edns_info edns;
};

/* Global array used by put_response and get_response to record
   response data.  The test DNS server returns the index of the array
   element which contains the actual response data.  This enables the
   test case to return arbitrary amounts of data with the limited
   number of bits which fit into an IP addres.

   The volatile specifier is needed because the test case accesses
   these variables from a callback function called from a function
   which is marked as __THROW (i.e., a leaf function which actually is
   not).  */
static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
static struct response_data ** volatile response_data_array;
volatile static size_t response_data_count;

/* Extract information from the query, store it in a struct
   response_data object, and return its index in the
   response_data_array.  */
static unsigned int
put_response (const struct resolv_response_context *ctx,
                 const char *qname, uint16_t qtype)
{
  xpthread_mutex_lock (&mutex);
  ++response_data_count;
  /* We only can represent 2**24 indexes in 10.0.0.0/8.  */
  TEST_VERIFY (response_data_count < (1 << 24));
  response_data_array = xrealloc
    (response_data_array, sizeof (*response_data_array) * response_data_count);
  unsigned int index = response_data_count - 1;
  struct response_data *data = xmalloc (sizeof (*data));
  *data = (struct response_data)
    {
      .qname = xstrdup (qname),
      .qtype = qtype,
      .edns = ctx->edns,
    };
  response_data_array[index] = data;
  xpthread_mutex_unlock (&mutex);
  return index;
}

/* Verify the index into the response_data array and return the data
   at it.  */
static struct response_data *
get_response (unsigned int index)
{
  xpthread_mutex_lock (&mutex);
  TEST_VERIFY_EXIT (index < response_data_count);
  struct response_data *result = response_data_array[index];
  xpthread_mutex_unlock (&mutex);
  return result;
}

/* Deallocate all response data.  */
static void
free_response_data (void)
{
  xpthread_mutex_lock (&mutex);
  size_t count = response_data_count;
  struct response_data **array = response_data_array;
  for (unsigned int i = 0; i < count; ++i)
    {
      struct response_data *data = array[i];
      free (data->qname);
      free (data);
    }
  free (array);
  response_data_array = NULL;
  response_data_count = 0;
  xpthread_mutex_unlock (&mutex);
}

#define EDNS_PROBE_EXAMPLE "edns-probe.example"

static void
response (const struct resolv_response_context *ctx,
          struct resolv_response_builder *b,
          const char *qname, uint16_t qclass, uint16_t qtype)
{
  TEST_VERIFY_EXIT (qname != NULL);

  const char *qname_compare = qname;

  /* The "formerr." prefix can be used to request a FORMERR response on the
     first server.  */
  bool send_formerr;
  if (strncmp ("formerr.", qname, strlen ("formerr.")) == 0)
    {
      send_formerr = true;
      qname_compare = qname + strlen ("formerr.");
    }
  else
    {
      send_formerr = false;
      qname_compare = qname;
    }

  /* The "tcp." prefix can be used to request TCP fallback.  */
  bool force_tcp;
  if (strncmp ("tcp.", qname_compare, strlen ("tcp.")) == 0)
    {
      force_tcp = true;
      qname_compare += strlen ("tcp.");
    }
  else
    force_tcp = false;

  enum {edns_probe} requested_qname;
  if (strcmp (qname_compare, EDNS_PROBE_EXAMPLE) == 0)
    requested_qname = edns_probe;
  else
    {
      support_record_failure ();
      printf ("error: unexpected QNAME: %s (reduced: %s)\n",
              qname, qname_compare);
      return;
    }
  TEST_VERIFY_EXIT (qclass == C_IN);
  struct resolv_response_flags flags = { };
  flags.tc = force_tcp && !ctx->tcp;
  if (!flags.tc && send_formerr && ctx->server_index == 0)
    /* Send a FORMERR for the first full response from the first
       server.  */
    flags.rcode = 1;          /* FORMERR */
  resolv_response_init (b, flags);
  resolv_response_add_question (b, qname, qclass, qtype);
  if (flags.tc || flags.rcode != 0)
    return;

  if (test_verbose)
    printf ("info: edns=%d payload_size=%d\n",
            ctx->edns.active, ctx->edns.payload_size);

  /* Encode the response_data object in multiple address records.
     Each record carries two bytes of payload data, and an index.  */
  resolv_response_section (b, ns_s_an);
  switch (requested_qname)
    {
    case edns_probe:
      {
        unsigned int index = put_response (ctx, qname, qtype);
        switch (qtype)
          {
          case T_A:
            {
              uint32_t addr = htonl (0x0a000000 | index);
              resolv_response_open_record (b, qname, qclass, qtype, 0);
              resolv_response_add_data (b, &addr, sizeof (addr));
              resolv_response_close_record (b);
            }
            break;
          case T_AAAA:
            {
              char addr[16]
                = {0x20, 0x01, 0xd, 0xb8, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   index >> 16, index >> 8, index};
              resolv_response_open_record (b, qname, qclass, qtype, 0);
              resolv_response_add_data (b, &addr, sizeof (addr));
              resolv_response_close_record (b);
            }
          }
      }
      break;
    }
}

/* Update *DATA with data from ADDRESS of SIZE.  Set the corresponding
   flag in SHADOW for each byte written.  */
static struct response_data *
decode_address (const void *address, size_t size)
{
  switch (size)
    {
    case 4:
      TEST_VERIFY (memcmp (address, "\x0a", 1) == 0);
      break;
    case 16:
      TEST_VERIFY (memcmp (address, "\x20\x01\x0d\xb8", 4) == 0);
      break;
    default:
      FAIL_EXIT1 ("unexpected address size %zu", size);
    }
  const unsigned char *addr = address;
  unsigned int index = addr[size - 3] * 256 * 256
    + addr[size - 2] * 256
    + addr[size - 1];
  return get_response (index);
}

static struct response_data *
decode_hostent (struct hostent *e)
{
  TEST_VERIFY_EXIT (e != NULL);
  TEST_VERIFY_EXIT (e->h_addr_list[0] != NULL);
  TEST_VERIFY (e->h_addr_list[1] == NULL);
  return decode_address (e->h_addr_list[0], e->h_length);
}

static struct response_data *
decode_addrinfo (struct addrinfo *ai, int family)
{
  struct response_data *data = NULL;
  while (ai != NULL)
    {
      if (ai->ai_family == family)
        {
          struct response_data *new_data;
          switch (family)
            {
            case AF_INET:
              {
                struct sockaddr_in *pin = (struct sockaddr_in *) ai->ai_addr;
                new_data = decode_address (&pin->sin_addr.s_addr, 4);
              }
              break;
            case AF_INET6:
              {
                struct sockaddr_in6 *pin = (struct sockaddr_in6 *) ai->ai_addr;
                new_data = decode_address (&pin->sin6_addr.s6_addr, 16);
              }
              break;
            default:
              FAIL_EXIT1 ("invalid address family %d", ai->ai_family);
            }
          if (data == NULL)
            data = new_data;
          else
            /* Check pointer equality because this should be the same
               response (same index).  */
            TEST_VERIFY (data == new_data);
        }
      ai = ai->ai_next;
    }
  TEST_VERIFY_EXIT (data != NULL);
  return data;
}

/* Updated by the main test loop in accordance with what is set in
   _res.options.  */
static bool use_edns;
static bool use_dnssec;

/* Verify the decoded response data against the flags above.  */
static void
verify_response_data_payload (struct response_data *data,
                              size_t expected_payload)
{
  bool edns = use_edns || use_dnssec;
  TEST_VERIFY (data->edns.active == edns);
  if (!edns)
    expected_payload = 0;
  if (data->edns.payload_size != expected_payload)
    {
      support_record_failure ();
      printf ("error: unexpected payload size %d (edns=%d)\n",
              (int) data->edns.payload_size, edns);
    }
  uint16_t expected_flags = 0;
  if (use_dnssec)
    expected_flags |= 0x8000;   /* DO flag.  */
  if (data->edns.flags != expected_flags)
    {
      support_record_failure ();
      printf ("error: unexpected EDNS flags 0x%04x (edns=%d)\n",
              (int) data->edns.flags, edns);
    }
}

/* Same as verify_response_data_payload, but use the default
   payload.  */
static void
verify_response_data (struct response_data *data)
{
  verify_response_data_payload (data, 1200);
}

static void
check_hostent (struct hostent *e)
{
  TEST_VERIFY_EXIT (e != NULL);
  verify_response_data (decode_hostent (e));
}

static void
do_ai (int family)
{
  struct addrinfo hints = { .ai_family = family };
  struct addrinfo *ai;
  int ret = getaddrinfo (EDNS_PROBE_EXAMPLE, "80", &hints, &ai);
  TEST_VERIFY_EXIT (ret == 0);
  switch (family)
    {
    case AF_INET:
    case AF_INET6:
      verify_response_data (decode_addrinfo (ai, family));
      break;
    case AF_UNSPEC:
      verify_response_data (decode_addrinfo (ai, AF_INET));
      verify_response_data (decode_addrinfo (ai, AF_INET6));
      break;
    default:
      FAIL_EXIT1 ("invalid address family %d", family);
    }
  freeaddrinfo (ai);
}

enum res_op
{
  res_op_search,
  res_op_query,
  res_op_querydomain,
  res_op_nsearch,
  res_op_nquery,
  res_op_nquerydomain,

  res_op_last = res_op_nquerydomain,
};

static const char *
res_op_string (enum res_op op)
{
  switch (op)
    {
      case res_op_search:
        return "res_search";
      case res_op_query:
        return "res_query";
      case res_op_querydomain:
        return "res_querydomain";
      case res_op_nsearch:
        return "res_nsearch";
      case res_op_nquery:
        return "res_nquery";
      case res_op_nquerydomain:
        return "res_nquerydomain";
    }
  FAIL_EXIT1 ("invalid res_op value %d", (int) op);
}

/* Call libresolv function OP to look up PROBE_NAME, with an answer
   buffer of SIZE bytes.  Check that the advertised UDP buffer size is
   in fact EXPECTED_BUFFER_SIZE.  */
static void
do_res_search (const char *probe_name, enum res_op op, size_t size,
               size_t expected_buffer_size)
{
  if (test_verbose)
    printf ("info: testing %s with buffer size %zu\n",
            res_op_string (op), size);
  unsigned char *buffer = xmalloc (size);
  int ret = -1;
  switch (op)
    {
    case res_op_search:
      ret = res_search (probe_name, C_IN, T_A, buffer, size);
      break;
    case res_op_query:
      ret = res_query (probe_name, C_IN, T_A, buffer, size);
      break;
    case res_op_nsearch:
      ret = res_nsearch (&_res, probe_name, C_IN, T_A, buffer, size);
      break;
    case res_op_nquery:
      ret = res_nquery (&_res, probe_name, C_IN, T_A, buffer, size);
      break;
    case res_op_querydomain:
    case res_op_nquerydomain:
      {
        char *example_stripped = xstrdup (probe_name);
        char *dot_example = strstr (example_stripped, ".example");
        if (dot_example != NULL && strcmp (dot_example, ".example") == 0)
          {
            /* Truncate the domain name.  */
            *dot_example = '\0';
            if (op == res_op_querydomain)
              ret = res_querydomain
              (example_stripped, "example", C_IN, T_A, buffer, size);
            else
              ret = res_nquerydomain
                (&_res, example_stripped, "example", C_IN, T_A, buffer, size);
          }
        else
          FAIL_EXIT1 ("invalid probe name: %s", probe_name);
        free (example_stripped);
      }
      break;
    }
  TEST_VERIFY_EXIT (ret > 12);
  unsigned char *end = buffer + ret;

  HEADER *hd = (HEADER *) buffer;
  TEST_VERIFY (ntohs (hd->qdcount) == 1);
  TEST_VERIFY (ntohs (hd->ancount) == 1);
  /* Skip over the header.  */
  unsigned char *p = buffer + sizeof (*hd);
  /* Skip over the question.  */
  ret = dn_skipname (p, end);
  TEST_VERIFY_EXIT (ret > 0);
  p += ret;
  TEST_VERIFY_EXIT (end - p >= 4);
  p += 4;
  /* Skip over the RNAME and the RR header, but stop at the RDATA
     length.  */
  ret = dn_skipname (p, end);
  TEST_VERIFY_EXIT (ret > 0);
  p += ret;
  TEST_VERIFY_EXIT (end - p >= 2 + 2 + 4 + 2 + 4);
  p += 2 + 2 + 4;
  /* The IP address should be 4 bytes long.  */
  TEST_VERIFY_EXIT (p[0] == 0);
  TEST_VERIFY_EXIT (p[1] == 4);
  /* Extract the address information.   */
  p += 2;
  struct response_data *data = decode_address (p, 4);

  verify_response_data_payload (data, expected_buffer_size);

  free (buffer);
}

static void
run_test (const char *probe_name)
{
  if (test_verbose)
    printf ("\ninfo: * use_edns=%d use_dnssec=%d\n",
            use_edns, use_dnssec);
  check_hostent (gethostbyname (probe_name));
  check_hostent (gethostbyname2 (probe_name, AF_INET));
  check_hostent (gethostbyname2 (probe_name, AF_INET6));
  do_ai (AF_UNSPEC);
  do_ai (AF_INET);
  do_ai (AF_INET6);

  for (int op = 0; op <= res_op_last; ++op)
    {
      do_res_search (probe_name, op, 301, 512);
      do_res_search (probe_name, op, 511, 512);
      do_res_search (probe_name, op, 512, 512);
      do_res_search (probe_name, op, 513, 513);
      do_res_search (probe_name, op, 657, 657);
      do_res_search (probe_name, op, 1199, 1199);
      do_res_search (probe_name, op, 1200, 1200);
      do_res_search (probe_name, op, 1201, 1200);
      do_res_search (probe_name, op, 65535, 1200);
    }
}

static int
do_test (void)
{
  for (int do_edns = 0; do_edns < 2; ++do_edns)
    for (int do_dnssec = 0; do_dnssec < 2; ++do_dnssec)
      for (int do_tcp = 0; do_tcp < 2; ++do_tcp)
        for (int do_formerr = 0; do_formerr < 2; ++do_formerr)
          {
            struct resolv_test *aux = resolv_test_start
              ((struct resolv_redirect_config)
               {
                 .response_callback = response,
               });

            use_edns = do_edns;
            if (do_edns)
              _res.options |= RES_USE_EDNS0;
            use_dnssec = do_dnssec;
            if (do_dnssec)
              _res.options |= RES_USE_DNSSEC;

            char *probe_name = xstrdup (EDNS_PROBE_EXAMPLE);
            if (do_tcp)
              {
                char *n = xasprintf ("tcp.%s", probe_name);
                free (probe_name);
                probe_name = n;
              }
            if (do_formerr)
              {
                /* Send a garbage query in an attempt to trigger EDNS
                   fallback.  */
                char *n = xasprintf ("formerr.%s", probe_name);
                gethostbyname (n);
                free (n);
              }

            run_test (probe_name);

            free (probe_name);
            resolv_test_end (aux);
          }

  free_response_data ();
  return 0;
}

#include <support/test-driver.c>
