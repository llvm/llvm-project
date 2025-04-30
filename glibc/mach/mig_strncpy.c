/* Silly pointless function MiG needs.  */

#include <mach.h>
#include <string.h>

vm_size_t
__mig_strncpy (char *dst, const char *src, vm_size_t len)
{
  return __stpncpy (dst, src, len) - dst;
}
weak_alias (__mig_strncpy, mig_strncpy)
