// RUN: %clang_cc1 -fsyntax-only -Wno-error=int-conversion -fnullability-safety -I %S/Inputs %s -verify=plain
// RUN: %clang_cc1 -fsyntax-only -Wno-error=int-conversion -fnullability-safety -fnullability-default=nonnull -DPARTIAL_HEADER -I %S/Inputs %s -verify=plain
// RUN: %clang_cc1 -fsyntax-only -Wno-error=int-conversion -fnullability-safety -fnullability-default=nullable -DPARTIAL_HEADER -I %S/Inputs %s -verify=tagged

// Only -fnullability-default=nullable tags unannotated pointers with
// _Null_unspecified; in the other modes the tag would not change any result,
// so types print as written. Leaving pointers untagged under the nonnull
// default must not turn on -Wnullability-completeness for the header.

#ifdef PARTIAL_HEADER
#include "nullability-safety-partially-annotated.h"
#endif

void param(int *q) {
  char c = q; // plain-warning{{incompatible pointer to integer conversion initializing 'char' with an expression of type 'int *'}} tagged-warning{{incompatible pointer to integer conversion initializing 'char' with an expression of type 'int * _Null_unspecified'}}
  (void)c;
}

void local(void) {
  int x = 0;
  int *l = &x;
  char c = l; // plain-warning{{incompatible pointer to integer conversion initializing 'char' with an expression of type 'int *'}} tagged-warning{{incompatible pointer to integer conversion initializing 'char' with an expression of type 'int * _Null_unspecified'}}
  (void)c;
}
