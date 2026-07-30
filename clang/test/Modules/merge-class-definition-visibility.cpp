// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fmodule-map-file=%S/Inputs/merge-class-definition-visibility/modmap \
// RUN:            -I%S/Inputs/merge-class-definition-visibility \
// RUN:            -fmodules-cache-path=%t %s -verify \
// RUN:            -fmodules-local-submodule-visibility
// expected-no-diagnostics

#include "c.h"
template<typename T> struct X { T t; };
typedef X<A> XA;
struct B;

#include "e.h"
// Ensure that this triggers the import of the second definition from e.h,
// which is necessary to make the definition of A visible in the template
// instantiation.
XA xa;

// Ensure that we make the definition of B visible. We made the parse-merged
// definition from e.h visible, which makes the definition from d.h visible,
// and that definition was merged into the canonical definition from b.h,
// so that becomes visible, and we have a visible definition.
B b;
