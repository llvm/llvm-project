#ifndef _IFADDRS_H
#include <inet/ifaddrs.h>

# ifndef _ISOMAC

#include <stdbool.h>
#include <stdint.h>

libc_hidden_proto (getifaddrs)
libc_hidden_proto (freeifaddrs)

extern int __getifaddrs (struct ifaddrs **__ifap);
libc_hidden_proto (__getifaddrs)
extern void __freeifaddrs (struct ifaddrs *__ifa);
libc_hidden_proto (__freeifaddrs)

struct in6addrinfo
{
  enum {
    in6ai_deprecated = 1,
    in6ai_homeaddress = 2
  } flags:8;
  uint8_t prefixlen;
  uint16_t :16;
  uint32_t index;
  uint32_t addr[4];
};

extern void __check_pf (bool *seen_ipv4, bool *seen_ipv6,
			struct in6addrinfo **in6ai, size_t *in6ailen)
  attribute_hidden;
extern void __free_in6ai (struct in6addrinfo *in6ai) attribute_hidden;
extern void __check_native (uint32_t a1_index, int *a1_native,
			    uint32_t a2_index, int *a2_native)
  attribute_hidden;

#if IS_IN (nscd)
extern uint32_t __bump_nl_timestamp (void) attribute_hidden;
#endif

# endif /* !_ISOMAC */
#endif	/* ifaddrs.h */
