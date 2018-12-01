// RUN: %clang_cc1 -fsyntax-only -fobjc-arc -Wno-objc-root-class -std=c++11 -ast-dump %s | FileCheck %s

// CHECK-NOT: ImplicitValueInitExpr

@interface I
@end

template <typename T>
void decode(I *p) {
  for (I *k in p) {}
}

void decode(I *p) {
  decode<int>(p);
}
