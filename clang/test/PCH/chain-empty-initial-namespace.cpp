// no PCH
// RUN: %clang_cc1 -include %s -include %s -fsyntax-only %s
// full PCH
// RUN: %clang_cc1 -chain-include %s -chain-include %s -fsyntax-only %s
#if !defined(PASS1)
#define PASS1

namespace foo {} // no external storage

#elif !defined(PASS2)
#define PASS2

namespace foo {
  void bar();
}

#else
// PASS3

void test() {
  foo::bar(); // no-error
}

#endif
