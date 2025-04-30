/* DNS test framework and libresolv redirection.
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

#include <support/resolv_test.h>

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <nss.h>
#include <resolv.h>
#include <search.h>
#include <stdlib.h>
#include <string.h>
#include <support/check.h>
#include <support/namespace.h>
#include <support/support.h>
#include <support/test-driver.h>
#include <support/xsocket.h>
#include <support/xthread.h>
#include <support/xunistd.h>
#include <sys/uio.h>
#include <unistd.h>

/* Response builder.  */

enum
  {
    max_response_length = 65536
  };

/* Used for locating domain names containing for the purpose of
   forming compression references.  */
struct compressed_name
{
  uint16_t offset;
  unsigned char length;
  unsigned char name[];         /* Without terminating NUL.  */
};

static struct compressed_name *
allocate_compressed_name (const unsigned char *encoded, unsigned int offset)
{
  /* Compute the length of the domain name.  */
  size_t length;
  {
    const unsigned char *p;
    for (p = encoded; *p != '\0';)
      {
        /* No compression references are allowed.  */
        TEST_VERIFY (*p <= 63);
        /* Skip over the label.  */
        p += 1 + *p;
      }
    length = p - encoded;
    ++length;                   /* For the terminating NUL byte.  */
  }
  TEST_VERIFY_EXIT (length <= 255);

  struct compressed_name *result
    = xmalloc (offsetof (struct compressed_name, name) + length);
  result->offset = offset;
  result->length = length;
  memcpy (result->name, encoded, length);
  return result;
}

/* Convert CH to lower case.  Only change letters in the ASCII
   range.  */
static inline unsigned char
ascii_tolower (unsigned char ch)
{
  if ('A' <= ch && ch <= 'Z')
    return ch - 'A' + 'a';
  else
    return ch;
}

/* Compare both names, for use with tsearch.  The order is arbitrary,
   but the comparison is case-insenstive.  */
static int
compare_compressed_name (const void *left, const void *right)
{
  const struct compressed_name *crleft = left;
  const struct compressed_name *crright = right;

  if (crleft->length != crright->length)
    /* The operands are converted to int before the subtraction.  */
    return crleft->length - crright->length;

  const unsigned char *nameleft = crleft->name;
  const unsigned char *nameright = crright->name;

  while (true)
    {
      int lenleft = *nameleft++;
      int lenright = *nameright++;

      /* Labels must not e compression references.  */
      TEST_VERIFY (lenleft <= 63);
      TEST_VERIFY (lenright <= 63);

      if (lenleft != lenright)
        return left - right;
      if (lenleft == 0)
        /* End of name reached without spotting a difference.  */
        return 0;
      /* Compare the label in a case-insenstive manner.  */
      const unsigned char *endnameleft = nameleft + lenleft;
      while (nameleft < endnameleft)
        {
          int l = *nameleft++;
          int r = *nameright++;
          if (l != r)
            {
              l = ascii_tolower (l);
              r = ascii_tolower (r);
              if (l != r)
                return l - r;
            }
        }
    }
}

struct resolv_response_builder
{
  const unsigned char *query_buffer;
  size_t query_length;

  size_t offset;                /* Bytes written so far in buffer.  */
  ns_sect section;              /* Current section in the DNS packet.  */
  unsigned int truncate_bytes;  /* Bytes to remove at end of response. */
  bool drop;                    /* Discard generated response.  */
  bool close;                   /* Close TCP client connection.  */

  /* Offset of the two-byte RDATA length field in the currently
     written RDATA sub-structure.  0 if no RDATA is being written.  */
  size_t current_rdata_offset;

  /* tsearch tree for locating targets for label compression.  */
  void *compression_offsets;

  /* Must be last.  Not zeroed for performance reasons.  */
  unsigned char buffer[max_response_length];
};

/* Response builder. */

void
resolv_response_init (struct resolv_response_builder *b,
                      struct resolv_response_flags flags)
{
  if (b->offset > 0)
    FAIL_EXIT1 ("response_init: called at offset %zu", b->offset);
  if (b->query_length < 12)
    FAIL_EXIT1 ("response_init called for a query of size %zu",
                b->query_length);
  if (flags.rcode > 15)
    FAIL_EXIT1 ("response_init: invalid RCODE %u", flags.rcode);

  /* Copy the transaction ID.  */
  b->buffer[0] = b->query_buffer[0];
  b->buffer[1] = b->query_buffer[1];

  /* Initialize the flags.  */
  b->buffer[2] = 0x80;                       /* Mark as response.   */
  b->buffer[2] |= b->query_buffer[2] & 0x01; /* Copy the RD bit.  */
  if (flags.tc)
    b->buffer[2] |= 0x02;
  b->buffer[3] = flags.rcode;
  if (!flags.clear_ra)
    b->buffer[3] |= 0x80;
  if (flags.ad)
    b->buffer[3] |= 0x20;

  /* Fill in the initial section count values.  */
  b->buffer[4] = flags.qdcount >> 8;
  b->buffer[5] = flags.qdcount;
  b->buffer[6] = flags.ancount >> 8;
  b->buffer[7] = flags.ancount;
  b->buffer[8] = flags.nscount >> 8;
  b->buffer[9] = flags.nscount;
  b->buffer[10] = flags.adcount >> 8;
  b->buffer[11] = flags.adcount;

  b->offset = 12;
}

void
resolv_response_section (struct resolv_response_builder *b, ns_sect section)
{
  if (b->offset == 0)
    FAIL_EXIT1 ("resolv_response_section: response_init not called before");
  if (section < b->section)
    FAIL_EXIT1 ("resolv_response_section: cannot go back to previous section");
  b->section = section;
}

/* Add a single byte to B.  */
static inline void
response_add_byte (struct resolv_response_builder *b, unsigned char ch)
{
  if (b->offset == max_response_length)
    FAIL_EXIT1 ("DNS response exceeds 64 KiB limit");
  b->buffer[b->offset] = ch;
  ++b->offset;
}

/* Add a 16-bit word VAL to B, in big-endian format.  */
static void
response_add_16 (struct resolv_response_builder *b, uint16_t val)
{
  response_add_byte (b, val >> 8);
  response_add_byte (b, val);
}

/* Increment the pers-section record counter in the packet header.  */
static void
response_count_increment (struct resolv_response_builder *b)
{
  unsigned int offset = b->section;
  offset = 4 + 2 * offset;
  ++b->buffer[offset + 1];
  if (b->buffer[offset + 1] == 0)
    {
      /* Carry.  */
      ++b->buffer[offset];
      if (b->buffer[offset] == 0)
        /* Overflow.  */
        FAIL_EXIT1 ("too many records in section");
    }
}

void
resolv_response_add_question (struct resolv_response_builder *b,
                              const char *name, uint16_t class, uint16_t type)
{
  if (b->offset == 0)
    FAIL_EXIT1 ("resolv_response_add_question: "
                "resolv_response_init not called");
  if (b->section != ns_s_qd)
    FAIL_EXIT1 ("resolv_response_add_question: "
                "must be called in the question section");

  resolv_response_add_name (b, name);
  response_add_16 (b, type);
  response_add_16 (b, class);

  response_count_increment (b);
}

void
resolv_response_add_name (struct resolv_response_builder *b,
                          const char *const origname)
{
  unsigned char encoded_name[NS_MAXDNAME];
  if (ns_name_pton (origname, encoded_name, sizeof (encoded_name)) < 0)
    FAIL_EXIT1 ("ns_name_pton (\"%s\"): %m", origname);

  /* Copy the encoded name into the output buffer, apply compression
     where possible.  */
  for (const unsigned char *name = encoded_name; ;)
    {
      if (*name == '\0')
        {
          /* We have reached the end of the name.  Add the terminating
             NUL byte.  */
          response_add_byte (b, '\0');
          break;
        }

      /* Set to the compression target if compression is possible.  */
      struct compressed_name *crname_target;

      /* Compression references can only reach the beginning of the
         packet.  */
      enum { compression_limit = 1 << 12 };

      {
        /* The trailing part of the name to be looked up in the tree
           with the compression targets.  */
        struct compressed_name *crname
          = allocate_compressed_name (name, b->offset);

        if (b->offset < compression_limit)
          {
            /* Add the name to the tree, for future compression
               references.  */
            void **ptr = tsearch (crname, &b->compression_offsets,
                                  compare_compressed_name);
            if (ptr == NULL)
              FAIL_EXIT1 ("tsearch out of memory");
            crname_target = *ptr;

            if (crname_target != crname)
              /* The new name was not actually added to the tree.
                 Deallocate it.  */
              free (crname);
            else
              /* Signal that the tree did not yet contain the name,
                 but keep the allocation because it is now part of the
                 tree.  */
              crname_target = NULL;
          }
        else
          {
            /* This name cannot be reached by a compression reference.
               No need to add it to the tree for future reference.  */
            void **ptr = tfind (crname, &b->compression_offsets,
                                compare_compressed_name);
            if (ptr != NULL)
              crname_target = *ptr;
            else
              crname_target = NULL;
            TEST_VERIFY (crname_target != crname);
            /* Not added to the tree.  */
            free (crname);
          }
      }

      if (crname_target != NULL)
        {
          /* The name is known.  Reference the previous location.  */
          unsigned int old_offset = crname_target->offset;
          TEST_VERIFY_EXIT (old_offset < compression_limit);
          response_add_byte (b, 0xC0 | (old_offset >> 8));
          response_add_byte (b, old_offset);
          break;
        }
      else
        {
          /* The name is new.  Add this label.  */
          unsigned int len = 1 + *name;
          resolv_response_add_data (b, name, len);
          name += len;
        }
    }
}

void
resolv_response_open_record (struct resolv_response_builder *b,
                             const char *name,
                             uint16_t class, uint16_t type, uint32_t ttl)
{
  if (b->section == ns_s_qd)
    FAIL_EXIT1 ("resolv_response_open_record called in question section");
  if (b->current_rdata_offset != 0)
    FAIL_EXIT1 ("resolv_response_open_record called with open record");

  resolv_response_add_name (b, name);
  response_add_16 (b, type);
  response_add_16 (b, class);
  response_add_16 (b, ttl >> 16);
  response_add_16 (b, ttl);

  b->current_rdata_offset = b->offset;
  /* Add room for the RDATA length.  */
  response_add_16 (b, 0);
}


void
resolv_response_close_record (struct resolv_response_builder *b)
{
  size_t rdata_offset = b->current_rdata_offset;
  if (rdata_offset == 0)
    FAIL_EXIT1 ("response_close_record called without open record");
  size_t rdata_length = b->offset - rdata_offset - 2;
  if (rdata_length > 65535)
    FAIL_EXIT1 ("RDATA length %zu exceeds limit", rdata_length);
  b->buffer[rdata_offset] = rdata_length >> 8;
  b->buffer[rdata_offset + 1] = rdata_length;
  response_count_increment (b);
  b->current_rdata_offset = 0;
}

void
resolv_response_add_data (struct resolv_response_builder *b,
                          const void *data, size_t length)
{
  size_t remaining = max_response_length - b->offset;
  if (remaining < length)
    FAIL_EXIT1 ("resolv_response_add_data: not enough room for %zu bytes",
                length);
  memcpy (b->buffer + b->offset, data, length);
  b->offset += length;
}

void
resolv_response_drop (struct resolv_response_builder *b)
{
  b->drop = true;
}

void
resolv_response_close (struct resolv_response_builder *b)
{
  b->close = true;
}

void
resolv_response_truncate_data (struct resolv_response_builder *b, size_t count)
{
  if (count > 65535)
    FAIL_EXIT1 ("resolv_response_truncate_data: argument too large: %zu",
                count);
  b->truncate_bytes = count;
}


size_t
resolv_response_length (const struct resolv_response_builder *b)
{
  return b->offset;
}

unsigned char *
resolv_response_buffer (const struct resolv_response_builder *b)
{
  unsigned char *result = xmalloc (b->offset);
  memcpy (result, b->buffer, b->offset);
  return result;
}

struct resolv_response_builder *
resolv_response_builder_allocate (const unsigned char *query_buffer,
                                  size_t query_length)
{
  struct resolv_response_builder *b = xmalloc (sizeof (*b));
  memset (b, 0, offsetof (struct resolv_response_builder, buffer));
  b->query_buffer = query_buffer;
  b->query_length = query_length;
  return b;
}

void
resolv_response_builder_free (struct resolv_response_builder *b)
{
  tdestroy (b->compression_offsets, free);
  free (b);
}

/* DNS query processing. */

/* Data extracted from the question section of a DNS packet.  */
struct query_info
{
  char qname[MAXDNAME];
  uint16_t qclass;
  uint16_t qtype;
  struct resolv_edns_info edns;
};

/* Update *INFO from the specified DNS packet.  */
static void
parse_query (struct query_info *info,
             const unsigned char *buffer, size_t length)
{
  HEADER hd;
  _Static_assert (sizeof (hd) == 12, "DNS header size");
  if (length < sizeof (hd))
    FAIL_EXIT1 ("malformed DNS query: too short: %zu bytes", length);
  memcpy (&hd, buffer, sizeof (hd));

  if (ntohs (hd.qdcount) != 1)
    FAIL_EXIT1 ("malformed DNS query: wrong question count: %d",
                (int) ntohs (hd.qdcount));
  if (ntohs (hd.ancount) != 0)
    FAIL_EXIT1 ("malformed DNS query: wrong answer count: %d",
                (int) ntohs (hd.ancount));
  if (ntohs (hd.nscount) != 0)
    FAIL_EXIT1 ("malformed DNS query: wrong authority count: %d",
                (int) ntohs (hd.nscount));
  if (ntohs (hd.arcount) > 1)
    FAIL_EXIT1 ("malformed DNS query: wrong additional count: %d",
                (int) ntohs (hd.arcount));

  int ret = dn_expand (buffer, buffer + length, buffer + sizeof (hd),
                       info->qname, sizeof (info->qname));
  if (ret < 0)
    FAIL_EXIT1 ("malformed DNS query: cannot uncompress QNAME");

  /* Obtain QTYPE and QCLASS.  */
  size_t remaining = length - (12 + ret);
  struct
  {
    uint16_t qtype;
    uint16_t qclass;
  } qtype_qclass;
  if (remaining < sizeof (qtype_qclass))
    FAIL_EXIT1 ("malformed DNS query: "
                "query lacks QCLASS/QTYPE, QNAME: %s", info->qname);
  memcpy (&qtype_qclass, buffer + 12 + ret, sizeof (qtype_qclass));
  info->qclass = ntohs (qtype_qclass.qclass);
  info->qtype = ntohs (qtype_qclass.qtype);

  memset (&info->edns, 0, sizeof (info->edns));
  if (ntohs (hd.arcount) > 0)
    {
      /* Parse EDNS record.  */
      struct __attribute__ ((packed, aligned (1)))
      {
        uint8_t root;
        uint16_t rtype;
        uint16_t payload;
        uint8_t edns_extended_rcode;
        uint8_t edns_version;
        uint16_t flags;
        uint16_t rdatalen;
      } rr;
      _Static_assert (sizeof (rr) == 11, "EDNS record size");

      if (remaining < 4 + sizeof (rr))
        FAIL_EXIT1 ("mailformed DNS query: no room for EDNS record");
      memcpy (&rr, buffer + 12 + ret + 4, sizeof (rr));
      if (rr.root != 0)
        FAIL_EXIT1 ("malformed DNS query: invalid OPT RNAME: %d\n", rr.root);
      if (rr.rtype != htons (41))
        FAIL_EXIT1 ("malformed DNS query: invalid OPT type: %d\n",
                    ntohs (rr.rtype));
      info->edns.active = true;
      info->edns.extended_rcode = rr.edns_extended_rcode;
      info->edns.version = rr.edns_version;
      info->edns.flags = ntohs (rr.flags);
      info->edns.payload_size = ntohs (rr.payload);
    }
}


/* Main testing framework.  */

/* Per-server information.  One struct is allocated for each test
   server.  */
struct resolv_test_server
{
  /* Local address of the server.  UDP and TCP use the same port.  */
  struct sockaddr_in address;

  /* File descriptor of the UDP server, or -1 if this server is
     disabled.  */
  int socket_udp;

  /* File descriptor of the TCP server, or -1 if this server is
     disabled.  */
  int socket_tcp;

  /* Counter of the number of responses processed so far.  */
  size_t response_number;

  /* Thread handles for the server threads (if not disabled in the
     configuration).  */
  pthread_t thread_udp;
  pthread_t thread_tcp;
};

/* Main struct for keeping track of libresolv redirection and
   testing.  */
struct resolv_test
{
  /* After initialization, any access to the struct must be performed
     while this lock is acquired.  */
  pthread_mutex_t lock;

  /* Data for each test server. */
  struct resolv_test_server servers[resolv_max_test_servers];

  /* Used if config.single_thread_udp is true.  */
  pthread_t thread_udp_single;

  struct resolv_redirect_config config;
  bool termination_requested;
};

/* Function implementing a server thread.  */
typedef void (*thread_callback) (struct resolv_test *, int server_index);

/* Storage for thread-specific data, for passing to the
   thread_callback function.  */
struct thread_closure
{
  struct resolv_test *obj;      /* Current test object.  */
  thread_callback callback;     /* Function to call.  */
  int server_index;             /* Index of the implemented server.  */
};

/* Wrap response_callback as a function which can be passed to
   pthread_create.  */
static void *
thread_callback_wrapper (void *arg)
{
  struct thread_closure *closure = arg;
  closure->callback (closure->obj, closure->server_index);
  free (closure);
  return NULL;
}

/* Start a server thread for the specified SERVER_INDEX, implemented
   by CALLBACK.  */
static pthread_t
start_server_thread (struct resolv_test *obj, int server_index,
                     thread_callback callback)
{
  struct thread_closure *closure = xmalloc (sizeof (*closure));
  *closure = (struct thread_closure)
    {
      .obj = obj,
      .callback = callback,
      .server_index = server_index,
    };
  return xpthread_create (NULL, thread_callback_wrapper, closure);
}

/* Process one UDP query.  Return false if a termination requested has
   been detected.  */
static bool
server_thread_udp_process_one (struct resolv_test *obj, int server_index)
{
  unsigned char query[512];
  struct sockaddr_storage peer;
  socklen_t peerlen = sizeof (peer);
  size_t length = xrecvfrom (obj->servers[server_index].socket_udp,
                             query, sizeof (query), 0,
                             (struct sockaddr *) &peer, &peerlen);
  /* Check for termination.  */
  {
    bool termination_requested;
    xpthread_mutex_lock (&obj->lock);
    termination_requested = obj->termination_requested;
    xpthread_mutex_unlock (&obj->lock);
    if (termination_requested)
      return false;
  }


  struct query_info qinfo;
  parse_query (&qinfo, query, length);
  if (test_verbose > 0)
    {
      if (test_verbose > 1)
        printf ("info: UDP server %d: incoming query:"
                " %zd bytes, %s/%u/%u, tnxid=0x%02x%02x\n",
                server_index, length, qinfo.qname, qinfo.qclass, qinfo.qtype,
                query[0], query[1]);
      else
        printf ("info: UDP server %d: incoming query:"
                " %zd bytes, %s/%u/%u\n",
                server_index, length, qinfo.qname, qinfo.qclass, qinfo.qtype);
    }

  struct resolv_response_context ctx =
    {
      .test = obj,
      .client_address = &peer,
      .client_address_length = peerlen,
      .query_buffer = query,
      .query_length = length,
      .server_index = server_index,
      .tcp = false,
      .edns = qinfo.edns,
    };
  struct resolv_response_builder *b
    = resolv_response_builder_allocate (query, length);
  obj->config.response_callback
    (&ctx, b, qinfo.qname, qinfo.qclass, qinfo.qtype);

  if (b->drop)
    {
      if (test_verbose)
        printf ("info: UDP server %d: dropping response to %s/%u/%u\n",
                server_index, qinfo.qname, qinfo.qclass, qinfo.qtype);
    }
  else
    {
      if (test_verbose)
        {
          if (b->offset >= 12)
            printf ("info: UDP server %d: sending response:"
                    " %zu bytes, RCODE %d (for %s/%u/%u)\n",
                    ctx.server_index, b->offset, b->buffer[3] & 0x0f,
                    qinfo.qname, qinfo.qclass, qinfo.qtype);
          else
            printf ("info: UDP server %d: sending response: %zu bytes"
                    " (for %s/%u/%u)\n",
                    server_index, b->offset,
                    qinfo.qname, qinfo.qclass, qinfo.qtype);
          if (b->truncate_bytes > 0)
            printf ("info:    truncated by %u bytes\n", b->truncate_bytes);
        }
      resolv_response_send_udp (&ctx, b);
    }
  resolv_response_builder_free (b);
  return true;
}

void
resolv_response_send_udp (const struct resolv_response_context *ctx,
                          struct resolv_response_builder *b)
{
  TEST_VERIFY_EXIT (!ctx->tcp);
  size_t to_send = b->offset;
  if (to_send < b->truncate_bytes)
    to_send = 0;
  else
    to_send -= b->truncate_bytes;

  /* Ignore most errors here because the other end may have closed
     the socket.  */
  if (sendto (ctx->test->servers[ctx->server_index].socket_udp,
              b->buffer, to_send, 0,
              ctx->client_address, ctx->client_address_length) < 0)
    TEST_VERIFY_EXIT (errno != EBADF);
}

/* UDP thread_callback function.  Variant for one thread per
   server.  */
static void
server_thread_udp (struct resolv_test *obj, int server_index)
{
  while (server_thread_udp_process_one (obj, server_index))
    ;
}

/* Single-threaded UDP processing function, for the single_thread_udp
   case.  */
static void *
server_thread_udp_single (void *closure)
{
  struct resolv_test *obj = closure;

  struct pollfd fds[resolv_max_test_servers];
  for (int server_index = 0; server_index < resolv_max_test_servers;
       ++server_index)
    if (obj->config.servers[server_index].disable_udp)
      fds[server_index] = (struct pollfd) {.fd = -1};
    else
      {
        fds[server_index] = (struct pollfd)
          {
            .fd = obj->servers[server_index].socket_udp,
            .events = POLLIN
          };

        /* Make the socket non-blocking.  */
        int flags = fcntl (obj->servers[server_index].socket_udp, F_GETFL, 0);
        if (flags < 0)
          FAIL_EXIT1 ("fcntl (F_GETFL): %m");
        flags |= O_NONBLOCK;
        if (fcntl (obj->servers[server_index].socket_udp, F_SETFL, flags) < 0)
          FAIL_EXIT1 ("fcntl (F_SETFL): %m");
      }

  while (true)
    {
      xpoll (fds, resolv_max_test_servers, -1);
      for (int server_index = 0; server_index < resolv_max_test_servers;
           ++server_index)
        if (fds[server_index].revents != 0)
          {
            if (!server_thread_udp_process_one (obj, server_index))
              goto out;
            fds[server_index].revents = 0;
          }
    }

 out:
  return NULL;
}

/* Start the single UDP handler thread (for the single_thread_udp
   case).  */
static void
start_server_thread_udp_single (struct resolv_test *obj)
{
  obj->thread_udp_single
    = xpthread_create (NULL, server_thread_udp_single, obj);
}

/* Data describing a TCP client connect.  */
struct tcp_thread_closure
{
  struct resolv_test *obj;
  int server_index;
  int client_socket;
};

/* Read a complete DNS query packet.  If EOF_OK, an immediate
   end-of-file condition is acceptable.  */
static bool
read_fully (int fd, void *buf, size_t len, bool eof_ok)
{
  const void *const end = buf + len;
  while (buf < end)
    {
      ssize_t ret = read (fd, buf, end - buf);
      if (ret == 0)
        {
          if (!eof_ok)
            {
              support_record_failure ();
              printf ("error: unexpected EOF on TCP connection\n");
            }
          return false;
        }
      else if (ret < 0)
        {
          if (!eof_ok || errno != ECONNRESET)
            {
              support_record_failure ();
              printf ("error: TCP read: %m\n");
            }
          return false;
        }
      buf += ret;
      eof_ok = false;
    }
  return true;
}

/* Write an array of iovecs.  Terminate the process on failure.  */
static void
writev_fully (int fd, struct iovec *buffers, size_t count)
{
  while (count > 0)
    {
      /* Skip zero-length write requests.  */
      if (buffers->iov_len == 0)
        {
          ++buffers;
          --count;
          continue;
        }
      /* Try to rewrite the remaing buffers.  */
      ssize_t ret = writev (fd, buffers, count);
      if (ret < 0)
        FAIL_EXIT1 ("writev: %m");
      if (ret == 0)
        FAIL_EXIT1 ("writev: invalid return value zero");
      /* Find the buffers that were successfully written.  */
      while (ret > 0)
        {
          if (count == 0)
            FAIL_EXIT1 ("internal writev consistency failure");
          /* Current buffer was partially written.  */
          if (buffers->iov_len > (size_t) ret)
            {
              buffers->iov_base += ret;
              buffers->iov_len -= ret;
              ret = 0;
            }
          else
            {
              ret -= buffers->iov_len;
              buffers->iov_len = 0;
              ++buffers;
              --count;
            }
        }
    }
}

/* Thread callback for handling a single established TCP connection to
   a client.  */
static void *
server_thread_tcp_client (void *arg)
{
  struct tcp_thread_closure *closure = arg;

  while (true)
    {
      /* Read packet length.  */
      uint16_t query_length;
      if (!read_fully (closure->client_socket,
                       &query_length, sizeof (query_length), true))
        break;
      query_length = ntohs (query_length);

      /* Read the packet.  */
      unsigned char *query_buffer = xmalloc (query_length);
      read_fully (closure->client_socket, query_buffer, query_length, false);

      struct query_info qinfo;
      parse_query (&qinfo, query_buffer, query_length);
      if (test_verbose > 0)
        {
          if (test_verbose > 1)
            printf ("info: UDP server %d: incoming query:"
                    " %d bytes, %s/%u/%u, tnxid=0x%02x%02x\n",
                    closure->server_index, query_length,
                    qinfo.qname, qinfo.qclass, qinfo.qtype,
                    query_buffer[0], query_buffer[1]);
          else
            printf ("info: TCP server %d: incoming query:"
                    " %u bytes, %s/%u/%u\n",
                    closure->server_index, query_length,
                    qinfo.qname, qinfo.qclass, qinfo.qtype);
        }

      struct resolv_response_context ctx =
        {
          .test = closure->obj,
          .query_buffer = query_buffer,
          .query_length = query_length,
          .server_index = closure->server_index,
          .tcp = true,
          .edns = qinfo.edns,
        };
      struct resolv_response_builder *b
        = resolv_response_builder_allocate (query_buffer, query_length);
      closure->obj->config.response_callback
        (&ctx, b, qinfo.qname, qinfo.qclass, qinfo.qtype);

      if (b->drop)
        {
          if (test_verbose)
            printf ("info: TCP server %d: dropping response to %s/%u/%u\n",
                    closure->server_index,
                    qinfo.qname, qinfo.qclass, qinfo.qtype);
        }
      else
        {
          if (test_verbose)
            printf ("info: TCP server %d: sending response: %zu bytes"
                    " (for %s/%u/%u)\n",
                    closure->server_index, b->offset,
                    qinfo.qname, qinfo.qclass, qinfo.qtype);
          uint16_t length = htons (b->offset);
          size_t to_send = b->offset;
          if (to_send < b->truncate_bytes)
            to_send = 0;
          else
            to_send -= b->truncate_bytes;
          struct iovec buffers[2] =
            {
              {&length, sizeof (length)},
              {b->buffer, to_send}
            };
          writev_fully (closure->client_socket, buffers, 2);
        }
      bool close_flag = b->close;
      resolv_response_builder_free (b);
      free (query_buffer);
      if (close_flag)
        break;
    }

  xclose (closure->client_socket);
  free (closure);
  return NULL;
}

/* thread_callback for the TCP case.  Accept connections and create a
   new thread for each client.  */
static void
server_thread_tcp (struct resolv_test *obj, int server_index)
{
  while (true)
    {
      /* Get the client conenction.  */
      int client_socket = xaccept
        (obj->servers[server_index].socket_tcp, NULL, NULL);

      /* Check for termination.  */
      xpthread_mutex_lock (&obj->lock);
      if (obj->termination_requested)
        {
          xpthread_mutex_unlock (&obj->lock);
          xclose (client_socket);
          break;
        }
      xpthread_mutex_unlock (&obj->lock);

      /* Spawn a new thread for handling this connection.  */
      struct tcp_thread_closure *closure = xmalloc (sizeof (*closure));
      *closure = (struct tcp_thread_closure)
        {
          .obj = obj,
          .server_index = server_index,
          .client_socket = client_socket,
        };

      pthread_t thr
        = xpthread_create (NULL, server_thread_tcp_client, closure);
      /* TODO: We should keep track of this thread so that we can
         block in resolv_test_end until it has exited.  */
      xpthread_detach (thr);
    }
}

/* Create UDP and TCP server sockets.  */
static void
make_server_sockets (struct resolv_test_server *server)
{
  while (true)
    {
      server->socket_udp = xsocket (AF_INET, SOCK_DGRAM, IPPROTO_UDP);
      server->socket_tcp = xsocket (AF_INET, SOCK_STREAM, IPPROTO_TCP);

      /* Pick the address for the UDP socket.  */
      server->address = (struct sockaddr_in)
        {
          .sin_family = AF_INET,
          .sin_addr = {.s_addr = htonl (INADDR_LOOPBACK)}
        };
      xbind (server->socket_udp,
             (struct sockaddr *)&server->address, sizeof (server->address));

      /* Retrieve the address. */
      socklen_t addrlen = sizeof (server->address);
      xgetsockname (server->socket_udp,
                    (struct sockaddr *)&server->address, &addrlen);

      /* Bind the TCP socket to the same address.  */
      {
        int on = 1;
        xsetsockopt (server->socket_tcp, SOL_SOCKET, SO_REUSEADDR,
                     &on, sizeof (on));
      }
      if (bind (server->socket_tcp,
                (struct sockaddr *)&server->address,
                sizeof (server->address)) != 0)
        {
          /* Port collision.  The UDP bind succeeded, but the TCP BIND
             failed.  We assume here that the kernel will pick the
             next local UDP address randomly.  */
          if (errno == EADDRINUSE)
            {
              xclose (server->socket_udp);
              xclose (server->socket_tcp);
              continue;
            }
          FAIL_EXIT1 ("TCP bind: %m");
        }
      xlisten (server->socket_tcp, 5);
      break;
    }
}

/* Like make_server_sockets, but the caller supplies the address to
   use.  */
static void
make_server_sockets_for_address (struct resolv_test_server *server,
                                 const struct sockaddr *addr)
{
  server->socket_udp = xsocket (AF_INET, SOCK_DGRAM, IPPROTO_UDP);
  server->socket_tcp = xsocket (AF_INET, SOCK_STREAM, IPPROTO_TCP);

  if (addr->sa_family == AF_INET)
    server->address = *(const struct sockaddr_in *) addr;
  else
    /* We cannot store the server address in the socket.  This should
       not matter if disable_redirect is used.  */
    server->address = (struct sockaddr_in) { .sin_family = 0, };

  xbind (server->socket_udp,
         (struct sockaddr *)&server->address, sizeof (server->address));
  xbind (server->socket_tcp,
         (struct sockaddr *)&server->address, sizeof (server->address));
  xlisten (server->socket_tcp, 5);
}

/* One-time initialization of NSS.  */
static void
resolv_redirect_once (void)
{
  /* Only use nss_dns.  */
  __nss_configure_lookup ("hosts", "dns");
  __nss_configure_lookup ("networks", "dns");
  /* Enter a network namespace for isolation and firewall state
     cleanup.  The tests will still work if these steps fail, but they
     may be less reliable.  */
  support_become_root ();
  support_enter_network_namespace ();
}
pthread_once_t resolv_redirect_once_var = PTHREAD_ONCE_INIT;

void
resolv_test_init (void)
{
  /* Perform one-time initialization of NSS.  */
  xpthread_once (&resolv_redirect_once_var, resolv_redirect_once);
}

/* Copy the search path from CONFIG.search to the _res object.  */
static void
set_search_path (struct resolv_redirect_config config)
{
  memset (_res.defdname, 0, sizeof (_res.defdname));
  memset (_res.dnsrch, 0, sizeof (_res.dnsrch));

  char *current = _res.defdname;
  char *end = current + sizeof (_res.defdname);

  for (unsigned int i = 0;
       i < sizeof (config.search) / sizeof (config.search[0]); ++i)
    {
      if (config.search[i] == NULL)
        continue;

      size_t length = strlen (config.search[i]) + 1;
      size_t remaining = end - current;
      TEST_VERIFY_EXIT (length <= remaining);
      memcpy (current, config.search[i], length);
      _res.dnsrch[i] = current;
      current += length;
    }
}

struct resolv_test *
resolv_test_start (struct resolv_redirect_config config)
{
  /* Apply configuration defaults.  */
  if (config.nscount == 0)
    config.nscount = resolv_max_test_servers;

  struct resolv_test *obj = xmalloc (sizeof (*obj));
  *obj = (struct resolv_test) {
    .config = config,
    .lock = PTHREAD_MUTEX_INITIALIZER,
  };

  if (!config.disable_redirect)
    resolv_test_init ();

  /* Create all the servers, to reserve the necessary ports.  */
  for (int server_index = 0; server_index < config.nscount; ++server_index)
    if (config.disable_redirect && config.server_address_overrides != NULL)
      make_server_sockets_for_address
        (obj->servers + server_index,
         config.server_address_overrides[server_index]);
    else
      make_server_sockets (obj->servers + server_index);

  /* Start server threads.  Disable the server ports, as
     requested.  */
  for (int server_index = 0; server_index < config.nscount; ++server_index)
    {
      struct resolv_test_server *server = obj->servers + server_index;
      if (config.servers[server_index].disable_udp)
        {
          xclose (server->socket_udp);
          server->socket_udp = -1;
        }
      else if (!config.single_thread_udp)
        server->thread_udp = start_server_thread (obj, server_index,
                                                  server_thread_udp);
      if (config.servers[server_index].disable_tcp)
        {
          xclose (server->socket_tcp);
          server->socket_tcp = -1;
        }
      else
        server->thread_tcp = start_server_thread (obj, server_index,
                                                  server_thread_tcp);
    }
  if (config.single_thread_udp)
    start_server_thread_udp_single (obj);

  if (config.disable_redirect)
    return obj;

  int timeout = 1;

  /* Initialize libresolv.  */
  TEST_VERIFY_EXIT (res_init () == 0);

  /* Disable IPv6 name server addresses.  The code below only
     overrides the IPv4 addresses.  */
  __res_iclose (&_res, true);
  _res._u._ext.nscount = 0;

  /* Redirect queries to the server socket.  */
  if (test_verbose)
    {
      printf ("info: old timeout value: %d\n", _res.retrans);
      printf ("info: old retry attempt value: %d\n", _res.retry);
      printf ("info: old _res.options: 0x%lx\n", _res.options);
      printf ("info: old _res.nscount value: %d\n", _res.nscount);
      printf ("info: old _res.ndots value: %d\n", _res.ndots);
    }
  _res.retrans = timeout;
  _res.retry = 4;
  _res.nscount = config.nscount;
  _res.options = RES_INIT | RES_RECURSE | RES_DEFNAMES | RES_DNSRCH;
  _res.ndots = 1;
  if (test_verbose)
    {
      printf ("info: new timeout value: %d\n", _res.retrans);
      printf ("info: new retry attempt value: %d\n", _res.retry);
      printf ("info: new _res.options: 0x%lx\n", _res.options);
      printf ("info: new _res.nscount value: %d\n", _res.nscount);
      printf ("info: new _res.ndots value: %d\n", _res.ndots);
    }
  for (int server_index = 0; server_index < config.nscount; ++server_index)
    {
      TEST_VERIFY_EXIT (obj->servers[server_index].address.sin_port != 0);
      _res.nsaddr_list[server_index] = obj->servers[server_index].address;
      if (test_verbose)
        {
          char buf[256];
          TEST_VERIFY_EXIT
            (inet_ntop (AF_INET, &obj->servers[server_index].address.sin_addr,
                        buf, sizeof (buf)) != NULL);
          printf ("info: server %d: %s/%u\n",
                  server_index, buf,
                  htons (obj->servers[server_index].address.sin_port));
        }
    }

  set_search_path (config);

  return obj;
}

void
resolv_test_end (struct resolv_test *obj)
{
  res_close ();

  xpthread_mutex_lock (&obj->lock);
  obj->termination_requested = true;
  xpthread_mutex_unlock (&obj->lock);

  /* Send trigger packets to unblock the server threads.  */
  for (int server_index = 0; server_index < obj->config.nscount;
       ++server_index)
    {
      if (!obj->config.servers[server_index].disable_udp)
        {
          int sock = xsocket (AF_INET, SOCK_DGRAM, IPPROTO_UDP);
          xsendto (sock, "", 1, 0,
                   (struct sockaddr *) &obj->servers[server_index].address,
                   sizeof (obj->servers[server_index].address));
          xclose (sock);
        }
      if (!obj->config.servers[server_index].disable_tcp)
        {
          int sock = xsocket (AF_INET, SOCK_STREAM, IPPROTO_TCP);
          xconnect (sock,
                    (struct sockaddr *) &obj->servers[server_index].address,
                    sizeof (obj->servers[server_index].address));
          xclose (sock);
        }
    }

  if (obj->config.single_thread_udp)
    xpthread_join (obj->thread_udp_single);

  /* Wait for the server threads to terminate.  */
  for (int server_index = 0; server_index < obj->config.nscount;
       ++server_index)
    {
      if (!obj->config.servers[server_index].disable_udp)
        {
          if (!obj->config.single_thread_udp)
            xpthread_join (obj->servers[server_index].thread_udp);
          xclose (obj->servers[server_index].socket_udp);
        }
      if (!obj->config.servers[server_index].disable_tcp)
        {
          xpthread_join (obj->servers[server_index].thread_tcp);
          xclose (obj->servers[server_index].socket_tcp);
        }
    }

  free (obj);
}
