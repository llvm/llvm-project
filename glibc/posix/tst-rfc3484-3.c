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

struct sockaddr_in addrs[] =
{
  { .sin_family = AF_INET, .sin_addr = { h (0xa0a86d1d) } },
  { .sin_family = AF_INET, .sin_addr = { h (0xa0a85d03) } },
  { .sin_family = AF_INET, .sin_addr = { h (0xa0a82c3d) } },
  { .sin_family = AF_INET, .sin_addr = { h (0xa0a86002) } },
  { .sin_family = AF_INET, .sin_addr = { h (0xa0a802f3) } },
  { .sin_family = AF_INET, .sin_addr = { h (0xa0a80810) } },
  { .sin_family = AF_INET, .sin_addr = { h (0xa0a85e02) } },
  { .sin_family = AF_INET, .sin_addr = { h (0xac162311) } },
  { .sin_family = AF_INET, .sin_addr = { h (0x0a324572) } }
};
#define naddrs (sizeof (addrs) / sizeof (addrs[0]))
static struct addrinfo ais[naddrs];
static struct sort_result results[naddrs];
static size_t order[naddrs];

static const int expected[naddrs] =
  {
    8, 0, 1, 2, 3, 4, 5, 6, 7
  };

static const struct scopeentry new_scopes[] =
  {
    { { { 169, 254, 0, 0 } }, h (0xffff0000), 2 },
    { { { 127, 0, 0, 0 } }, h (0xff000000), 2 },
    { { { 10, 0, 0, 0 } }, h (0xff000000), 5 },
    { { { 192, 168, 0, 0 } }, h(0xffff0000), 5 },
    { { { 0, 0, 0, 0 } }, h (0x00000000), 14 }
  };


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
  scopes= new_scopes;

  struct sockaddr_in so;
  so.sin_family = AF_INET;
  so.sin_addr.s_addr = h (0x0aa85f19);
  /* Clear the rest of the structure to avoid warnings.  */
  memset (so.sin_zero, '\0', sizeof (so.sin_zero));

  for (int i = 0; i < naddrs; ++i)
    {
      ais[i].ai_family = AF_INET;
      ais[i].ai_addr = (struct sockaddr *) &addrs[i];
      results[i].dest_addr = &ais[i];
      results[i].got_source_addr = true;
      memcpy(&results[i].source_addr, &so, sizeof (so));
      results[i].source_addr_len = sizeof (so);
      results[i].source_addr_flags = 0;
      results[i].prefixlen = 8;
      results[i].index = 0;

      order[i] = i;
    }

  struct sort_result_combo combo = { .results = results, .nresults = naddrs };
  qsort_r (order, naddrs, sizeof (order[0]), rfc3484_sort, &combo);

  int result = 0;
  for (int i = 0; i < naddrs; ++i)
    {
      struct in_addr addr = ((struct sockaddr_in *) (results[order[i]].dest_addr->ai_addr))->sin_addr;

      int here = memcmp (&addr, &addrs[expected[i]].sin_addr,
			 sizeof (struct in_addr));
      printf ("[%d] = %s: %s\n", i, inet_ntoa (addr), here ? "FAIL" : "OK");
      result |= here;
    }

  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
