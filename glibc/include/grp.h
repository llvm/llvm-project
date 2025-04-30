#ifndef _GRP_H
#include <grp/grp.h>

#ifndef _ISOMAC
libc_hidden_proto (setgroups)

/* Now define the internal interfaces.  */
extern int __getgrent_r (struct group *__resultbuf, char *buffer,
			 size_t __buflen, struct group **__result)
     attribute_hidden;
extern int __old_getgrent_r (struct group *__resultbuf, char *buffer,
			     size_t __buflen, struct group **__result);
extern int __fgetgrent_r (FILE * __stream, struct group *__resultbuf,
			  char *buffer, size_t __buflen,
			  struct group **__result) attribute_hidden;

/* Search for an entry with a matching group ID.  */
extern int __getgrgid_r (__gid_t __gid, struct group *__resultbuf,
			 char *__buffer, size_t __buflen,
			 struct group **__result) attribute_hidden;
extern int __old_getgrgid_r (__gid_t __gid, struct group *__resultbuf,
			     char *__buffer, size_t __buflen,
			     struct group **__result);

/* Search for an entry with a matching group name.  */
extern int __getgrnam_r (const char *__name, struct group *__resultbuf,
			 char *__buffer, size_t __buflen,
			 struct group **__result) attribute_hidden;
extern int __old_getgrnam_r (const char *__name, struct group *__resultbuf,
			     char *__buffer, size_t __buflen,
			     struct group **__result);

#define DECLARE_NSS_PROTOTYPES(service)					   \
extern enum nss_status _nss_ ## service ## _setgrent (int);		   \
extern enum nss_status _nss_ ## service ## _endgrent (void);		   \
extern enum nss_status _nss_ ## service ## _getgrgid_r			   \
		       (gid_t gid, struct group *grp, char *buffer,	   \
			size_t buflen, int *errnop);			   \
extern enum nss_status _nss_ ## service ## _getgrnam_r			   \
		       (const char *name, struct group *grp,		   \
			char *buffer, size_t buflen, int *errnop);	   \
extern enum nss_status _nss_ ## service ##_getgrent_r			   \
		       (struct group *result, char *buffer, size_t buflen, \
			int *errnop);					   \
extern enum nss_status _nss_ ## service ##_initgroups_dyn		   \
		       (const char *user, gid_t group, long int *start,	   \
			long int *size, gid_t **groupsp, long int limit,   \
			int *errnop);

DECLARE_NSS_PROTOTYPES (compat)
DECLARE_NSS_PROTOTYPES (files)
DECLARE_NSS_PROTOTYPES (hesiod)

#undef DECLARE_NSS_PROTOTYPES
#endif
#endif
