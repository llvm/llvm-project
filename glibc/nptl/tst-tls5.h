#include <stdint.h>
#include <stddef.h>

struct tls_obj
{
  const char *name;
  uintptr_t addr;
  size_t size;
  size_t align;
};
extern struct tls_obj tls_registry[];

#define TLS_REGISTER(x)				\
static void __attribute__((constructor))	\
tls_register_##x (void)				\
{						\
  size_t i;					\
  for (i = 0; tls_registry[i].name; ++i);	\
  tls_registry[i].name = #x;			\
  tls_registry[i].addr = (uintptr_t) &x;	\
  tls_registry[i].size = sizeof (x);		\
  tls_registry[i].align = __alignof__ (x);	\
}
