#ifndef _RESOLV_H_

# ifndef _ISOMAC
#  include <stdbool.h>
#  define RES_SET_H_ERRNO(r,x)			\
  do						\
    {						\
      (r)->res_h_errno = x;			\
      __set_h_errno(x);				\
    }						\
  while (0)
# endif

#include <resolv/resolv.h>

# if defined _RESOLV_H_ && !defined _ISOMAC

# if IS_IN (libc)
#  define __resp __libc_resp
# endif
extern __thread struct __res_state *__resp attribute_tls_model_ie;
# undef _res
# define _res (*__resp)

/* Now define the internal interfaces.  */
extern int __res_vinit (res_state, int) attribute_hidden;
extern void _sethtent (int);
extern struct hostent *_gethtent (void);
extern struct hostent *_gethtbyname (const char *__name);
extern struct hostent *_gethtbyname2 (const char *__name, int __af);
struct hostent *_gethtbyaddr (const char *addr, size_t __len, int __af);
extern uint32_t _getlong (const unsigned char *__src);
extern uint16_t _getshort (const unsigned char *__src);
extern void __res_iclose (res_state statp, bool free_addr);
libc_hidden_proto (__res_ninit)
libc_hidden_proto (__res_nclose)
libc_hidden_proto (__res_iclose)
libc_hidden_proto (__res_randomid)
libc_hidden_proto (__res_state)

libresolv_hidden_proto (_sethtent)
libresolv_hidden_proto (_gethtent)
libresolv_hidden_proto (_gethtbyaddr)
libresolv_hidden_proto (_gethtbyname2)
libresolv_hidden_proto (__putlong)
libresolv_hidden_proto (__putshort)
libresolv_hidden_proto (__p_cdnname)
libresolv_hidden_proto (__p_fqnname)
libresolv_hidden_proto (__p_option)
libresolv_hidden_proto (__sym_ntos)
libresolv_hidden_proto (__p_rcode)
libresolv_hidden_proto (__p_class)
libresolv_hidden_proto (__p_type)
libresolv_hidden_proto (__loc_ntoa)
libresolv_hidden_proto (__fp_nquery)
libresolv_hidden_proto (__fp_query)
libresolv_hidden_proto (__b64_ntop)
libresolv_hidden_proto (__dn_count_labels)

extern __typeof (dn_expand) __libc_dn_expand;
libc_hidden_proto (__libc_dn_expand)
extern __typeof (dn_skipname) __libc_dn_skipname;
libc_hidden_proto (__libc_dn_skipname)
extern __typeof (res_dnok) __libc_res_dnok;
libc_hidden_proto (__libc_res_dnok)
extern __typeof (res_hnok) __libc_res_hnok;
libc_hidden_proto (__libc_res_hnok)
extern __typeof (__res_nameinquery) __libc_res_nameinquery;
libc_hidden_proto (__libc_res_nameinquery)
extern __typeof (__res_queriesmatch) __libc_res_queriesmatch;
libc_hidden_proto (__libc_res_queriesmatch)

# endif /* _RESOLV_H_ && !_ISOMAC */
#endif
