// RUN: %clang_cc1 -emit-llvm-only %s
namespace A { }
namespace B = A;

namespace b {}

void foo() {
    namespace a = b;
}
