#include_next <ifreq.h>

static inline struct ifreq *
__if_nextreq (struct ifreq *ifr)
{
#ifdef _HAVE_SA_LEN
  if (ifr->ifr_addr.sa_len > sizeof ifr->ifr_addr)
    return (struct ifreq *) ((char *) &ifr->ifr_addr + ifr->ifr_addr.sa_len);
#endif
  return ifr + 1;
}

extern void __ifreq (struct ifreq **ifreqs, int *num_ifs, int sockfd)
     attribute_hidden;
