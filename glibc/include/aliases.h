#ifndef _ALIASES_H
#include <inet/aliases.h>

# ifndef _ISOMAC

extern int __getaliasent_r (struct aliasent *__restrict __result_buf,
			    char *__restrict __buffer, size_t __buflen,
			    struct aliasent **__restrict __result)
     attribute_hidden;
extern int __old_getaliasent_r (struct aliasent *__restrict __result_buf,
				char *__restrict __buffer, size_t __buflen,
				struct aliasent **__restrict __result);

extern int __getaliasbyname_r (const char *__restrict __name,
			       struct aliasent *__restrict __result_buf,
			       char *__restrict __buffer, size_t __buflen,
			       struct aliasent **__restrict __result)
     attribute_hidden;
extern int __old_getaliasbyname_r (const char *__restrict __name,
				   struct aliasent *__restrict __result_buf,
				   char *__restrict __buffer, size_t __buflen,
				   struct aliasent **__restrict __result);

#define DECLARE_NSS_PROTOTYPES(service)					     \
extern enum nss_status _nss_ ## service ## _setaliasent (void);		     \
extern enum nss_status _nss_ ## service ## _endaliasent (void);		     \
extern enum nss_status _nss_ ## service ## _getaliasent_r		     \
		       (struct aliasent *alias, char *buffer, size_t buflen, \
			int *errnop);					     \
extern enum nss_status _nss_ ## service ## _getaliasbyname_r		     \
		       (const char *name, struct aliasent *alias,	     \
			char *buffer, size_t buflen, int *errnop);


DECLARE_NSS_PROTOTYPES (files)
#undef DECLARE_NSS_PROTOTYPES

# endif /* !_ISOMAC */
#endif
