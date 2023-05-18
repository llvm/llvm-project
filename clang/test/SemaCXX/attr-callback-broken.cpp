// RUN: %clang_cc1 %s -verify -fsyntax-only

class C_in_class {
#define HAS_THIS
#include "../Sema/attr-callback-broken.c"
#undef HAS_THIS
};
