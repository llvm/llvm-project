// Test with pch.
// RUN: %clang_cc1 -emit-pch -std=c++17  -o %t %s
// RUN: %clang_cc1 -include-pch %t -emit-llvm -std=c++17 -o - %s

#ifndef HEADER
#define HEADER

namespace RP47219 {
typedef int MyInt;
template <typename T>
class Some {
 public:
  explicit Some(T, MyInt) {}
};

struct Foo {};
void ParseNatural() {
  Some(Foo(), 1);
}
}

#else

#endif
