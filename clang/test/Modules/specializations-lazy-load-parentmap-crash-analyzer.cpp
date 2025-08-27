// REQUIRES: staticanalyzer
//
// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file --leading-lines %s %t
//
// Prepare the BMIs.
// RUN: %clang_cc1 -std=c++20 -emit-module-interface -o %t/mod_a-part1.pcm %t/mod_a-part1.cppm
// RUN: %clang_cc1 -std=c++20 -emit-module-interface -o %t/mod_a-part2.pcm %t/mod_a-part2.cppm
// RUN: %clang_cc1 -std=c++20 -emit-module-interface -o %t/mod_a.pcm %t/mod_a.cppm -fmodule-file=mod_a:part2=%t/mod_a-part2.pcm -fmodule-file=mod_a:part1=%t/mod_a-part1.pcm
// RUN: %clang_cc1 -std=c++20 -emit-module-interface -o %t/mod_b.pcm %t/mod_b.cppm -fmodule-file=mod_a:part2=%t/mod_a-part2.pcm -fmodule-file=mod_a=%t/mod_a.pcm -fmodule-file=mod_a:part1=%t/mod_a-part1.pcm

// Trigger the construction of the parent map (which is necessary to trigger the bug this regression test is for) using ArrayBoundV2 checker:
// RUN: %clang_cc1 -std=c++20 -analyze -analyzer-checker=security,alpha.security -analyzer-output=text %t/test-array-bound-v2.cpp -fmodule-file=mod_a:part2=%t/mod_a-part2.pcm -fmodule-file=mod_a=%t/mod_a.pcm -fmodule-file=mod_a:part1=%t/mod_a-part1.pcm -fmodule-file=mod_b=%t/mod_b.pcm

//--- mod_a-part1.cppm
module;
namespace mod_a {
template <int> struct Important;
}

namespace mod_a {
Important<0>& instantiate1();
} // namespace mod_a
export module mod_a:part1;

export namespace mod_a {
using ::mod_a::instantiate1;
}

//--- mod_a-part2.cppm
module;
namespace mod_a {
template <int> struct Important;
}

namespace mod_a {
template <int N> Important<N>& instantiate2();
namespace part2InternalInstantiations {
// During the construction of the parent map, we iterate over ClassTemplateDecl::specializations() for 'Important'.
// After GH119333, the following instantiations get loaded between the call to spec_begin() and spec_end().
// This used to invalidate the begin iterator returned by spec_begin() by the time the end iterator is returned.
// This is a regression test for that.
Important<1> fn1();
Important<2> fn2();
Important<3> fn3();
Important<4> fn4();
Important<5> fn5();
Important<6> fn6();
Important<7> fn7();
Important<8> fn8();
Important<9> fn9();
Important<10> fn10();
Important<11> fn11();
}
} // namespace mod_a
export module mod_a:part2;

export namespace mod_a {
using ::mod_a::instantiate2;
}

//--- mod_a.cppm
export module mod_a;
export import :part1;
export import :part2;

//--- mod_b.cppm
export module mod_b;
import mod_a;

void a() {
  mod_a::instantiate1();
  mod_a::instantiate2<42>();
}

//--- test-array-bound-v2.cpp
import mod_b;

extern void someFunc(char* first, char* last);
void triggerParentMapContextCreationThroughArrayBoundV2() {
  // This code currently causes the ArrayBoundV2 checker to create the ParentMapContext.
  // Once it detects an access to buf[100], the checker looks through the parents to find '&' operator.
  // (this is needed since taking the address of past-the-end pointer is allowed by the checker)
  char buf[100];
  someFunc(&buf[0], &buf[100]);
}
