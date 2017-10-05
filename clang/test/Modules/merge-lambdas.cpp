// RUN: %clang_cc1 -std=c++11 -emit-llvm-only -fmodules %s

// PR33924: ensure that we merge together local lambas in multiple definitions
// of the same function.

#pragma clang module build format
module format {}
#pragma clang module contents
#pragma clang module begin format
struct A { template<typename T> void doFormat(T &&out) {} };
template<typename T> void format(T t) { A().doFormat([]{}); }
#pragma clang module end
#pragma clang module endbuild

#pragma clang module build foo1
module foo1 { export * }
#pragma clang module contents
#pragma clang module begin foo1
#pragma clang module import format
inline void foo1() {
  format(0);
}
#pragma clang module end
#pragma clang module endbuild

#pragma clang module build foo2
module foo2 { export * }
#pragma clang module contents
#pragma clang module begin foo2
#pragma clang module import format
inline void foo2() {
  format(0);
}
#pragma clang module end
#pragma clang module endbuild

#pragma clang module import foo1
#pragma clang module import foo2

int main() {
  foo1();
  foo2();
}
