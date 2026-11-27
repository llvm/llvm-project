// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/a.cppm -emit-reduced-module-interface -o %t/test-A.pcm
// RUN: %clang_cc1 -std=c++20 %t/N.cppm -emit-reduced-module-interface -o %t/test-N.pcm
// RUN: %clang_cc1 -std=c++20 %t/B.cppm -verify -fsyntax-only -fprebuilt-module-path=%t

//--- a.h
namespace N {
inline namespace impl   {
    template <typename>
    class C {
    template <typename> friend void foo();
    };

    template <typename> void foo() {}
} // namespace impl
} // namespace N

//--- a.cppm
// This is some unrelated file. It also #includes system headers, but
// here does not even export anything.
module;
#include "a.h"
export module test:A;
// To make sure they won't elided.
using N::C;
using N::foo;

//--- N.cppm
module;
#include "a.h"
export module test:N;

// Now wrap these names into a module and export them:
export {
  namespace N   {
    inline namespace impl    {
      using N::impl::C;
      using N::impl::foo;
    }
  }
}

//--- B.cppm
// expected-no-diagnostics
// A file that consumes the partitions from the other two files,
// including the exported N::C name.
module test:B;
import :N;
import :A;

N::C<int> x;
