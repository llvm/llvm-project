// RUN: %clang_cc1 -emit-llvm %s -o /dev/null

namespace A {
  typedef int B;
}
struct B {
};
using ::A::B;
