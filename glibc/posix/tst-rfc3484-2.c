#include <stdbool.h>
#include <stdio.h>
#include <ifaddrs.h>
#include <stdint.h>

/* Internal definitions used in the libc code.  */
#define __getservbyname_r getservbyname_r
#define __socket socket
#define __getsockname getsockname
#define __inet_aton inet_aton
#define __gethostbyaddr_r gethostbyaddr_r
#define __gethostbyname2_r gethostbyname2_r
#define __qsort_r qsort_r
#define __stat64 stat64

void
attribute_hidden
__check_pf (bool *p1, bool *p2, struct in6addrinfo **in6ai, size_t *in6ailen)
{
  *p1 = *p2 = true;
  *in6ai = NULL;
  *in6ailen = 0;
}

void
attribute_hidden
__free_in6ai (struct in6addrinfo *ai)
{
}

void
attribute_hidden
__check_native (uint32_t a1_index, int *a1_native,
		uint32_t a2_index, int *a2_native)
{
}

int
attribute_hidden
__idna_to_ascii_lz (const char *input, char **output, int flags)
{
  return 0;
}

int
attribute_hidden
__idna_to_unicode_lzlz (const char *input, char **output, int flags)
{
  *output = NULL;
  return 0;
}

void
attribute_hidden
_res_hconf_init (void)
{
}

#undef	USE_NSCD
#include "../sysdeps/posix/getaddrinfo.c"

nss_action_list __nss_hosts_database attribute_hidden;

/* This is the beginning of the real test code.  The above defines
   (among other things) the function rfc3484_sort.  */


#if __BYTE_ORDER == __BIG_ENDIAN
# define h(n) n
#else
# define h(n) __bswap_constant_32 (n)
#endif


ssize_t
__getline (char **lineptr, size_t *n, FILE *s)
{
  *lineptr = NULL;
  *n = 0;
  return 0;
}


static int
do_test (void)
{
  labels = default_labels;
  precedence = default_precedence;
  scopes = default_scopes;

  struct sockaddr_in so1;
  so1.sin_family = AF_INET;
  so1.sin_addr.s_addr = h (0xc0a85f19);
  /* Clear the rest of the structure to avoid warnings.  */
  memset (so1.sin_zero, '\0', sizeof (so1.sin_zero));

  struct sockaddr_in sa1;
  sa1.sin_family = AF_INET;
  sa1.sin_addr.s_addr = h (0xe0a85f19);

  struct addrinfo ai1;
  ai1.ai_family = AF_INET;
  ai1.ai_addr = (struct sockaddr *) &sa1;

  struct sockaddr_in6 so2;
  so2.sin6_family = AF_INET6;
  so2.sin6_addr.s6_addr32[0] = h (0xfec01234);
  so2.sin6_addr.s6_addr32[1] = 1;
  so2.sin6_addr.s6_addr32[2] = 1;
  so2.sin6_addr.s6_addr32[3] = 1;

  struct sockaddr_in6 sa2;
  sa2.sin6_family = AF_INET6;
  sa2.sin6_addr.s6_addr32[0] = h (0x07d10001);
  sa2.sin6_addr.s6_addr32[1] = 1;
  sa2.sin6_addr.s6_addr32[2] = 1;
  sa2.sin6_addr.s6_addr32[3] = 1;

  struct addrinfo ai2;
  ai2.ai_family = AF_INET6;
  ai2.ai_addr = (struct sockaddr *) &sa2;


  struct sort_result results[2];
  size_t order[2];

  results[0].dest_addr = &ai1;
  results[0].got_source_addr = true;
  results[0].source_addr_len = sizeof (so1);
  results[0].source_addr_flags = 0;
  results[0].prefixlen = 16;
  results[0].index = 0;
  memcpy (&results[0].source_addr, &so1, sizeof (so1));
  order[0] = 0;

  results[1].dest_addr = &ai2;
  results[1].got_source_addr = true;
  results[1].source_addr_len = sizeof (so2);
  results[1].source_addr_flags = 0;
  results[1].prefixlen = 16;
  results[1].index = 0;
  memcpy (&results[1].source_addr, &so2, sizeof (so2));
  order[1] = 1;


  struct sort_result_combo combo = { .results = results, .nresults = 2 };
  qsort_r (order, 2, sizeof (order[0]), rfc3484_sort, &combo);

  int result = 0;
  if (results[order[0]].dest_addr->ai_family == AF_INET6)
    {
      puts ("wrong order in first test");
      result |= 1;
    }


  /* And again, this time with the reverse starting order.  */
  results[1].dest_addr = &ai1;
  results[1].got_source_addr = true;
  results[1].source_addr_len = sizeof (so1);
  results[1].source_addr_flags = 0;
  results[1].prefixlen = 16;
  results[1].index = 0;
  memcpy (&results[1].source_addr, &so1, sizeof (so1));
  order[1] = 1;

  results[0].dest_addr = &ai2;
  results[0].got_source_addr = true;
  results[0].source_addr_len = sizeof (so2);
  results[0].source_addr_flags = 0;
  results[0].prefixlen = 16;
  results[0].index = 0;
  memcpy (&results[0].source_addr, &so2, sizeof (so2));
  order[0] = 0;


  qsort_r (order, 2, sizeof (order[0]), rfc3484_sort, &combo);

  if (results[order[0]].dest_addr->ai_family == AF_INET6)
    {
      puts ("wrong order in second test");
      result |= 1;
    }

  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
