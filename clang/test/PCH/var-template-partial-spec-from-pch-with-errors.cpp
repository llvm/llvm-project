// Build a PCH that contains compiler errors (an unresolved #include), then use
// it and force instantiation of a class template that was deserialized from it.
// The class has a member variable template with a partial specialization; when
// the enclosing class template comes from the deserialized PCH, the primary
// member variable template has not been instantiated into the current
// instantiation yet, so the lookup for it used to come back empty and crash
// (assertion "Instantiation found nothing?" in +Asserts builds, null
// dereference otherwise). See GH202956.

// RUN: %clang_cc1 -x c++-header -std=c++23 -fallow-pch-with-compiler-errors \
// RUN:   -emit-pch -o %t %S/var-template-partial-spec-from-pch-with-errors.h
// RUN: %clang_cc1 -std=c++23 -fallow-pch-with-compiler-errors -include-pch %t \
// RUN:   -fsyntax-only -verify %s

// expected-no-diagnostics

// Completing wrapper<int, int> instantiates the member variable template
// partial specialization above.
GH202956::wrapper<int, int> w;
