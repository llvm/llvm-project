#include <stdint.h>
#include <ptrcheck.h>

// both-note@+2{{passing argument to parameter 'p' here}}
// strict-note@+1{{passing argument to parameter 'p' here}}
static inline int * __single funcAdopted(int * __single p) {
  return p;
}

#pragma clang system_header

static inline int* funcSDK(intptr_t x) {
  if (x % 128)
    // both-error@+1{{non-pointer to safe pointer conversion is not allowed with -fbounds-safety; use '__unsafe_forge_single' or '__unsafe_forge_bidi_indexable'}}
    return funcAdopted(x);
  else
    // strict-error@+1{{passing 'int *' to parameter of incompatible type 'int *__single' casts away '__unsafe_indexable' qualifier; use '__unsafe_forge_single' or '__unsafe_forge_bidi_indexable' to perform this conversion}}
    return funcAdopted((int*)x);
}
