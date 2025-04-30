#ifndef _ARPA_NAMESER_H_

#include <resolv/arpa/nameser.h>

# ifndef _ISOMAC

/* If the machine allows unaligned access we can do better than using
   the NS_GET16, NS_GET32, NS_PUT16, and NS_PUT32 macros from the
   installed header.  */
#include <string.h>
#include <stdint.h>
#include <netinet/in.h>

extern const struct _ns_flagdata _ns_flagdata[] attribute_hidden;

#if _STRING_ARCH_unaligned

# undef NS_GET16
# define NS_GET16(s, cp) \
  do {									      \
    const uint16_t *t_cp = (const uint16_t *) (cp);			      \
    (s) = ntohs (*t_cp);						      \
    (cp) += NS_INT16SZ;							      \
  } while (0)

# undef NS_GET32
# define NS_GET32(l, cp) \
  do {									      \
    const uint32_t *t_cp = (const uint32_t *) (cp);			      \
    (l) = ntohl (*t_cp);						      \
    (cp) += NS_INT32SZ;							      \
  } while (0)

# undef NS_PUT16
# define NS_PUT16(s, cp) \
  do {									      \
    uint16_t *t_cp = (uint16_t *) (cp);					      \
    *t_cp = htons (s);							      \
    (cp) += NS_INT16SZ;							      \
  } while (0)

# undef NS_PUT32
# define NS_PUT32(l, cp) \
  do {									      \
    uint32_t *t_cp = (uint32_t *) (cp);					      \
    *t_cp = htonl (l);							      \
    (cp) += NS_INT32SZ;							      \
  } while (0)

#endif

extern unsigned int	__ns_get16 (const unsigned char *) __THROW;
extern unsigned long	__ns_get32 (const unsigned char *) __THROW;
int __ns_name_ntop (const unsigned char *, char *, size_t) __THROW;
int __ns_name_unpack (const unsigned char *, const unsigned char *,
		      const unsigned char *, unsigned char *, size_t) __THROW;

#define ns_msg_getflag(handle, flag) \
  (((handle)._flags & _ns_flagdata[flag].mask) >> _ns_flagdata[flag].shift)

libresolv_hidden_proto (ns_get16)
libresolv_hidden_proto (ns_get32)
libresolv_hidden_proto (ns_put16)
libresolv_hidden_proto (ns_put32)
libresolv_hidden_proto (ns_initparse)
libresolv_hidden_proto (ns_skiprr)
libresolv_hidden_proto (ns_parserr)
libresolv_hidden_proto (ns_sprintrr)
libresolv_hidden_proto (ns_sprintrrf)
libresolv_hidden_proto (ns_samedomain)
libresolv_hidden_proto (ns_format_ttl)

extern __typeof (ns_makecanon) __libc_ns_makecanon;
libc_hidden_proto (__libc_ns_makecanon)
extern __typeof (ns_name_compress) __ns_name_compress;
libc_hidden_proto (__ns_name_compress)
extern __typeof (ns_name_ntop) __ns_name_ntop;
libc_hidden_proto (__ns_name_ntop)
extern __typeof (ns_name_pack) __ns_name_pack;
libc_hidden_proto (__ns_name_pack)
extern __typeof (ns_name_pton) __ns_name_pton;
libc_hidden_proto (__ns_name_pton)
extern __typeof (ns_name_skip) __ns_name_skip;
libc_hidden_proto (__ns_name_skip)
extern __typeof (ns_name_uncompress) __ns_name_uncompress;
libc_hidden_proto (__ns_name_uncompress)
extern __typeof (ns_name_unpack) __ns_name_unpack;
libc_hidden_proto (__ns_name_unpack)
extern __typeof (ns_samename) __libc_ns_samename;
libc_hidden_proto (__libc_ns_samename)

# endif /* !_ISOMAC */
#endif
