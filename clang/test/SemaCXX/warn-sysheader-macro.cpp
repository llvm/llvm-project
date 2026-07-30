// RUN: %clang_cc1 -verify -fsyntax-only -Wshadow -Wold-style-cast -Wc++20-designator %s

// Test that macro expansions from system headers don't trigger 'syntactic'
// warnings that are not actionable.

#ifdef IS_SYSHEADER
#pragma clang system_header

#define SANITY(a) (a / 0)

#define SHADOW(a) __extension__({ int v = a; v; })

#define OLD_STYLE_CAST(a) ((int) (a))

struct Foo {
  int x;
};
#define DESIGNATED_INITIALIZERS (Foo{.x = 123})

#else

#define IS_SYSHEADER
#include __FILE__

void testSanity() {
  // Validate that the test is set up correctly
  int i = SANITY(0); // expected-warning {{division by zero is undefined}}
}

void PR16093() {
  // no -Wshadow in system macro expansion
  int i = SHADOW(SHADOW(1));
}

void PR18147() {
  // no -Wold-style-cast in system macro expansion
  int i = OLD_STYLE_CAST(0);
}

void PR52944() {
  // no -Wc++20-designator in system macro expansion
  auto i = DESIGNATED_INITIALIZERS;
}

#endif
