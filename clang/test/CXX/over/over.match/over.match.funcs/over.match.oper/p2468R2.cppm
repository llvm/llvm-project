// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/A.cppm -emit-module-interface -o %t/A.pcm
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t -I%t %t/p2468r2.cpp -verify

//--- A.cppm
module;
export module A;
export {
namespace NS {
struct S {};
bool operator==(S, int);
} // namespace NS
}

namespace NS { bool operator!=(S, int); } // Not visible.


//--- p2468r2.cpp
// expected-no-diagnostics
import A;
bool x = 0 == NS::S(); // Ok. operator!= from module A is not visible.
