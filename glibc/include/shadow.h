#ifndef _SHADOW_H
#include <shadow/shadow.h>

# ifndef _ISOMAC

/* Now define the internal interfaces.  */
extern int __getspent_r (struct spwd *__result_buf, char *__buffer,
			 size_t __buflen, struct spwd **__result)
     attribute_hidden;
extern int __old_getspent_r (struct spwd *__result_buf, char *__buffer,
			     size_t __buflen, struct spwd **__result);
extern int __getspnam_r (const char *__name, struct spwd *__result_buf,
			 char *__buffer, size_t __buflen,
			 struct spwd **__result) attribute_hidden;
extern int __old_getspnam_r (const char *__name, struct spwd *__result_buf,
			     char *__buffer, size_t __buflen,
			     struct spwd **__result);
extern int __sgetspent_r (const char *__string,
			  struct spwd *__result_buf, char *__buffer,
			  size_t __buflen, struct spwd **__result)
     attribute_hidden;
extern int __fgetspent_r (FILE *__stream, struct spwd *__result_buf,
			  char *__buffer, size_t __buflen,
			  struct spwd **__result) attribute_hidden;
extern int __lckpwdf (void);
extern int __ulckpwdf (void);

#define DECLARE_NSS_PROTOTYPES(service)					\
extern enum nss_status _nss_ ## service ## _setspent (int);		\
extern enum nss_status _nss_ ## service ## _endspent (void);		\
extern enum nss_status _nss_ ## service ## _getspent_r			\
		       (struct spwd *pwd, char *buffer, size_t buflen,	\
			int *errnop);					\
extern enum nss_status _nss_ ## service ## _getspnam_r			\
		       (const char *name, struct spwd *pwd,		\
			char *buffer, size_t buflen, int *errnop);

DECLARE_NSS_PROTOTYPES (compat)
DECLARE_NSS_PROTOTYPES (files)
DECLARE_NSS_PROTOTYPES (hesiod)

#undef DECLARE_NSS_PROTOTYPES


# endif /* !_ISOMAC */
#endif
