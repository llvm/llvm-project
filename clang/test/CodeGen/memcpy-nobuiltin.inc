#include <stddef.h>
extern void *memcpy(void *dest, void const *from, size_t n);

#ifdef WITH_DECL
inline void *memcpy(void *dest, void const *from, size_t n) {
  char const *ifrom = from;
  char *idest = dest;
  while (n--)
    *idest++ = *ifrom++;
  return dest;
}
#endif
#ifdef WITH_SELF_REFERENCE_DECL
inline void *memcpy(void *dest, void const *from, size_t n) {
  if (n != 0)
    memcpy(dest, from, n);
  return dest;
}
#endif
