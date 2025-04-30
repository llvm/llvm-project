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

#ifndef SUPPORT_RESOLV_TEST_H
#define SUPPORT_RESOLV_TEST_H

#include <arpa/nameser.h>
#include <stdbool.h>
#include <sys/cdefs.h>

__BEGIN_DECLS

/* Information about EDNS properties of a DNS query.  */
struct resolv_edns_info
{
  bool active;
  uint8_t extended_rcode;
  uint8_t version;
  uint16_t flags;
  uint16_t payload_size;
};

/* This opaque struct collects information about the resolver testing
   currently in progress.  */
struct resolv_test;

/* This struct provides context information when the response callback
   specified in struct resolv_redirect_config is invoked. */
struct resolv_response_context
{
  struct resolv_test *test;
  void *client_address;
  size_t client_address_length;
  unsigned char *query_buffer;
  size_t query_length;
  int server_index;
  bool tcp;
  struct resolv_edns_info edns;
};

/* Produces a deep copy of the context.  */
struct resolv_response_context *
  resolv_response_context_duplicate (const struct resolv_response_context *);

/* Frees the copy.  For the context passed to the response function,
   this happens implicitly.  */
void resolv_response_context_free (struct resolv_response_context *);

/* This opaque struct is used to construct responses from within the
   response callback function.  */
struct resolv_response_builder;

enum
  {
    /* Maximum number of test servers supported by the framework.  */
    resolv_max_test_servers = 3,
  };

/* Configuration settings specific to individual test servers.  */
struct resolv_redirect_server_config
{
  bool disable_tcp;             /* If true, no TCP server is listening.  */
  bool disable_udp;             /* If true, no UDP server is listening.  */
};

/* Instructions for setting up the libresolv redirection.  */
struct resolv_redirect_config
{
  /* The response_callback function is called for every incoming DNS
     packet, over UDP or TCP.  It must be specified, the other
     configuration settings are optional.  */
  void (*response_callback) (const struct resolv_response_context *,
                             struct resolv_response_builder *,
                             const char *qname,
                             uint16_t qclass, uint16_t qtype);

  /* Per-server configuration.  */
  struct resolv_redirect_server_config servers[resolv_max_test_servers];

  /* Search path entries.  The first entry serves as the default
     domain name as well.  */
  const char *search[7];

  /* Number of servers to activate in resolv.  0 means the default,
     resolv_max_test_servers.  */
  int nscount;

  /* If true, use a single thread to process all UDP queries.  This
     may results in more predictable ordering of queries and
     responses.  */
  bool single_thread_udp;

  /* Do not rewrite the _res variable or change NSS defaults.  Use
     server_address_overrides below to tell the testing framework on
     which addresses to create the servers.  */
  bool disable_redirect;

  /* Use these addresses for creating the DNS servers.  The array must
     have ns_count (or resolv_max_test_servers) sockaddr * elements if
     not NULL.  */
  const struct sockaddr *const *server_address_overrides;
};

/* Configure NSS to use, nss_dns only for aplicable databases, and try
   to put the process into a network namespace for better isolation.
   This may have to be called before resolv_test_start, before the
   process creates any threads.  Otherwise, initialization is
   performed by resolv_test_start implicitly.  */
void resolv_test_init (void);

/* Initiate resolver testing.  This updates the _res variable as
   needed.  As a side effect, NSS is reconfigured to use nss_dns only
   for aplicable databases, and the process may enter a network
   namespace for better isolation.  */
struct resolv_test *resolv_test_start (struct resolv_redirect_config);

/* Call this function at the end of resolver testing, to free
   resources and report pending errors (if any).  */
void resolv_test_end (struct resolv_test *);

/* The remaining facilities in this file are used for constructing
   response packets from the response_callback function.  */

/* Special settings for constructing responses from the callback.  */
struct resolv_response_flags
{
  /* 4-bit response code to incorporate into the response. */
  unsigned char rcode;

  /* If true, the TC (truncation) flag will be set.  */
  bool tc;

  /* If true, the AD (authenticated data) flag will be set.  */
  bool ad;

  /* If true, do not set the RA (recursion available) flag in the
     response.  */
  bool clear_ra;

  /* Initial section count values.  Can be used to artificially
     increase the counts, for malformed packet testing.*/
  unsigned short qdcount;
  unsigned short ancount;
  unsigned short nscount;
  unsigned short adcount;
};

/* Begin a new response with the requested flags.  Must be called
   first.  */
void resolv_response_init (struct resolv_response_builder *,
                           struct resolv_response_flags);

/* Switches to the section in the response packet.  Only forward
   movement is supported.  */
void resolv_response_section (struct resolv_response_builder *, ns_sect);

/* Add a question record to the question section.  */
void resolv_response_add_question (struct resolv_response_builder *,
                                   const char *name, uint16_t class,
                                   uint16_t type);
/* Starts a new resource record with the specified owner name, class,
   type, and TTL.  Data is supplied with resolv_response_add_data or
   resolv_response_add_name.  */
void resolv_response_open_record (struct resolv_response_builder *,
                                  const char *name, uint16_t class,
                                  uint16_t type, uint32_t ttl);

/* Add unstructed bytes to the RDATA part of a resource record.  */
void resolv_response_add_data (struct resolv_response_builder *,
                               const void *, size_t);

/* Add a compressed domain name to the RDATA part of a resource
   record.  */
void resolv_response_add_name (struct resolv_response_builder *,
                               const char *name);

/* Mark the end of the constructed record.  Must be called last.  */
void resolv_response_close_record (struct resolv_response_builder *);

/* Drop this query packet (that is, do not send a response, not even
   an empty packet).  */
void resolv_response_drop (struct resolv_response_builder *);

/* In TCP mode, close the connection after this packet (if a response
   is sent).  */
void resolv_response_close (struct resolv_response_builder *);

/* The size of the response packet built so far.  */
size_t resolv_response_length (const struct resolv_response_builder *);

/* Allocates a response builder tied to a specific query packet,
   starting at QUERY_BUFFER, containing QUERY_LENGTH bytes.  */
struct resolv_response_builder *
  resolv_response_builder_allocate (const unsigned char *query_buffer,
                                    size_t query_length);

/* Deallocates a response buffer.  */
void resolv_response_builder_free (struct resolv_response_builder *);

/* Sends a UDP response using a specific context.  This can be used to
   reorder or duplicate responses, along with
   resolv_response_context_duplicate and
   response_builder_allocate.  */
void resolv_response_send_udp (const struct resolv_response_context *,
                               struct resolv_response_builder *);

__END_DECLS

#endif /* SUPPORT_RESOLV_TEST_H */
