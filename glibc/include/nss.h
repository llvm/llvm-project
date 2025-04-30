#ifndef _NSS_H
#include <nss/nss.h>

#ifndef _ISOMAC

# include <stddef.h>
# include <stdint.h>

# define NSS_INVALID_FIELD_CHARACTERS ":\n"
extern const char __nss_invalid_field_characters[] attribute_hidden;

_Bool __nss_valid_field (const char *value) attribute_hidden;
_Bool __nss_valid_list_field (char **list)  attribute_hidden;
const char *__nss_rewrite_field (const char *value, char **to_be_freed)
  attribute_hidden;

/* Compute a hash value for LENGTH bytes starting at KEY.  This is the
   hash function used by the nscd for the cache mapping files.  */
uint32_t __nss_hash (const void *__key, size_t __length);
libc_hidden_proto (__nss_hash)

#endif /* !_ISOMAC */
#endif /* _NSS_H */
