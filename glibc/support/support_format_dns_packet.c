/* Convert a DNS packet to a human-readable representation.
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

#include <support/format_nss.h>

#include <arpa/inet.h>
#include <resolv.h>
#include <stdbool.h>
#include <support/check.h>
#include <support/support.h>
#include <support/xmemstream.h>

struct in_buffer
{
  const unsigned char *data;
  size_t size;
};

static inline bool
extract_8 (struct in_buffer *in, unsigned char *value)
{
  if (in->size == 0)
    return false;
  *value = in->data[0];
  ++in->data;
  --in->size;
  return true;
}

static inline bool
extract_16 (struct in_buffer *in, unsigned short *value)
{
  if (in->size < 2)
    return false;
  *value = (in->data[0] << 8) | in->data[1];
  in->data += 2;
  in->size -= 2;
  return true;
}

static inline bool
extract_32 (struct in_buffer *in, unsigned *value)
{
  if (in->size < 4)
    return false;
  unsigned a = in->data[0];
  unsigned b = in->data[1];
  unsigned c = in->data[2];
  unsigned d = in->data[3];
  *value = (a << 24) | (b << 16) | (c << 8) | d;
  in->data += 4;
  in->size -= 4;
  return true;
}

static inline bool
extract_bytes (struct in_buffer *in, size_t length, struct in_buffer *value)
{
  if (in->size < length)
    return false;
  *value = (struct in_buffer) {in->data, length};
  in->data += length;
  in->size -= length;
  return true;
}

struct dname
{
  char name[MAXDNAME + 1];
};

static bool
extract_name (struct in_buffer full, struct in_buffer *in, struct dname *value)
{
  const unsigned char *full_end = full.data + full.size;
  /* Sanity checks; these indicate buffer misuse.  */
  TEST_VERIFY_EXIT
    (!(in->data < full.data || in->data > full_end
       || in->size > (size_t) (full_end - in->data)));
  int ret = dn_expand (full.data, full_end, in->data,
                       value->name, sizeof (value->name));
  if (ret < 0)
    return false;
  in->data += ret;
  in->size -= ret;
  return true;
}

char *
support_format_dns_packet (const unsigned char *buffer, size_t length)
{
  struct in_buffer full = { buffer, length };
  struct in_buffer in = full;
  struct xmemstream mem;
  xopen_memstream (&mem);

  unsigned short txnid;
  unsigned short flags;
  unsigned short qdcount;
  unsigned short ancount;
  unsigned short nscount;
  unsigned short adcount;
  if (!(extract_16 (&in, &txnid)
        && extract_16 (&in, &flags)
        && extract_16 (&in, &qdcount)
        && extract_16 (&in, &ancount)
        && extract_16 (&in, &nscount)
        && extract_16 (&in, &adcount)))
    {
      fprintf (mem.out, "error: could not parse DNS header\n");
      goto out;
    }
  if (qdcount != 1)
    {
      fprintf (mem.out, "error: question count is %d, not 1\n", qdcount);
      goto out;
    }
  struct dname qname;
  if (!extract_name (full, &in, &qname))
    {
      fprintf (mem.out, "error: malformed QNAME\n");
      goto out;
    }
  unsigned short qtype;
  unsigned short qclass;
  if (!(extract_16 (&in, &qtype)
        && extract_16 (&in, &qclass)))
    {
      fprintf (mem.out, "error: malformed question\n");
      goto out;
    }
  if (qtype != T_A && qtype != T_AAAA && qtype != T_PTR)
    {
      fprintf (mem.out, "error: unsupported QTYPE %d\n", qtype);
      goto out;
    }

  fprintf (mem.out, "name: %s\n", qname.name);

  for (int i = 0; i < ancount; ++i)
    {
      struct dname rname;
      if (!extract_name (full, &in, &rname))
        {
          fprintf (mem.out, "error: malformed record name\n");
          goto out;
        }
      unsigned short rtype;
      unsigned short rclass;
      unsigned ttl;
      unsigned short rdlen;
      struct in_buffer rdata;
      if (!(extract_16 (&in, &rtype)
            && extract_16 (&in, &rclass)
            && extract_32 (&in, &ttl)
            && extract_16 (&in, &rdlen)
            && extract_bytes (&in, rdlen, &rdata)))
        {
          fprintf (mem.out, "error: malformed record header\n");
          goto out;
        }
      /* Skip non-matching record types.  */
      if ((rtype != qtype && rtype != T_CNAME) || rclass != qclass)
        continue;
      switch (rtype)
        {
        case T_A:
          if (rdlen == 4)
              fprintf (mem.out, "address: %d.%d.%d.%d\n",
                       rdata.data[0],
                       rdata.data[1],
                       rdata.data[2],
                       rdata.data[3]);
          else
            fprintf (mem.out, "error: A record of size %d: %s\n",
                     rdlen, rname.name);
          break;
        case T_AAAA:
          {
            if (rdlen == 16)
              {
                char buf[100];
                if (inet_ntop (AF_INET6, rdata.data, buf, sizeof (buf)) == NULL)
                  fprintf (mem.out, "error: AAAA record decoding failed: %m\n");
                else
                  fprintf (mem.out, "address: %s\n", buf);
              }
            else
              fprintf (mem.out, "error: AAAA record of size %d: %s\n",
                       rdlen, rname.name);
          }
          break;
        case T_CNAME:
        case T_PTR:
          {
            struct dname name;
            if (extract_name (full, &rdata, &name))
              fprintf (mem.out, "name: %s\n", name.name);
            else
              fprintf (mem.out, "error: malformed CNAME/PTR record\n");
          }
        }
    }

 out:
  xfclose_memstream (&mem);
  return mem.buffer;
}
