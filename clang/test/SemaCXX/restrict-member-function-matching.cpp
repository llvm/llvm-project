// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

struct S{ void a(); };

// GCC allows '__restrict' in the cv-qualifier-seq of member functions but
// ignores it for pretty much everything (except that the type of 'this' in
// is '__restrict' iff the *definition* is '__restrict' as well).
static_assert(__is_same(void (S::*) (), void (S::*) () __restrict));

namespace gh11039 {
class foo {
  int member[4];

  void bar(int * a);
};

void foo::bar(int * a) __restrict {
  member[3] = *a;
}
}