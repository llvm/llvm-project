// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/a.cppm -emit-reduced-module-interface -o %t/test-A.pcm
// RUN: %clang_cc1 -std=c++20 %t/N.cppm -emit-reduced-module-interface -o %t/test-N.pcm
// RUN: %clang_cc1 -std=c++20 %t/B.cppm -verify -fsyntax-only -fprebuilt-module-path=%t

//--- a.h
namespace N {

    template <typename>
    class C {
    template <typename> friend void foo();
    };

    template <typename> void foo() {}
} // namespace N

//--- a.cppm
// This is some unrelated file. It also #includes system headers, but
// here does not even export anything.
module;
#include "a.h"
export module test:A;
export {
    using N::C;
    using N::foo;
}

//--- std.h
// Declarations typically #included from C++ header files:
namespace N {               // In practice, this would be namespace std
    inline namespace impl {   // In practice, this would be namespace __1
        template <typename>
        class C {
        template <typename> friend void foo();
        };
    
        template <typename> void foo() {}
    } // namespace impl
    } // namespace N

//--- N.cppm
module;
#include "std.h"
export module test:N;

// Now wrap these names into a module and export them:
export {
    namespace N   {
        using N::C;
        using N::foo;
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
