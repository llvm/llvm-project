// RUN: %clang_cc1 -emit-llvm %s -o - | FileCheck %s
// The template should compile to linkonce linkage, not weak linkage.

// CHECK-NOT: weak
template<class T>
void thefunc();

template<class T>
inline void thefunc() {}

void test() {
  thefunc<int>();
}
