// Compound assignments to a __declspec(property) with a type-dependent operand
// must be represented by a CompoundAssignOperator; serializing them used to
// read the computation types past the end of a plain BinaryOperator.

// RUN: %clang_cc1 -std=c++20 -fdeclspec -ast-dump %s | FileCheck %s

// RUN: %clang_cc1 -std=c++20 -fdeclspec -x c++-header -emit-pch -o %t %s
// RUN: %clang_cc1 -std=c++20 -fdeclspec -include-pch %t -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++20 -fdeclspec -include-pch %t -ast-dump-all %s | FileCheck %s

// RUN: %clang_cc1 -std=c++20 -fdeclspec -x c++-header -emit-pch -fpch-instantiate-templates -o %t.inst %s
// RUN: %clang_cc1 -std=c++20 -fdeclspec -include-pch %t.inst -fsyntax-only -verify %s

// expected-no-diagnostics

#ifndef HEADER
#define HEADER

template <typename T>
struct V {
  T data[2];
  T get_x() const { return data[0]; }
  void put_x(T v) { data[0] = v; }
  __declspec(property(get = get_x, put = put_x)) T x;

  T get_at(int i) const { return data[i]; }
  void put_at(int i, T v) { data[i] = v; }
  __declspec(property(get = get_at, put = put_at)) T at[];

  template <typename U> void f(U s) {
    this->x += static_cast<T>(s);
    this->x -= static_cast<T>(s);
    this->x *= static_cast<T>(s);
    this->x /= static_cast<T>(s);
    this->x %= static_cast<T>(s);
    this->x <<= static_cast<T>(s);
    this->x >>= static_cast<T>(s);
    this->x &= static_cast<T>(s);
    this->x |= static_cast<T>(s);
    this->x ^= static_cast<T>(s);
    this->at[1] += static_cast<T>(s);
  }
};

// CHECK-LABEL: FunctionTemplateDecl {{.*}} f
// CHECK:       CXXMethodDecl {{.*}} f 'void (U)'
// CHECK:       CompoundAssignOperator {{.*}} '<dependent type>' {{.*}}'+=' ComputeLHSTy='<dependent type>' ComputeResultTy='<dependent type>'
// CHECK-NEXT:    MSPropertyRefExpr {{.*}} '<pseudo-object type>' lvalue
// CHECK:       CompoundAssignOperator {{.*}} '-='
// CHECK:       CompoundAssignOperator {{.*}} '*='
// CHECK:       CompoundAssignOperator {{.*}} '/='
// CHECK:       CompoundAssignOperator {{.*}} '%='
// CHECK:       CompoundAssignOperator {{.*}} '<<='
// CHECK:       CompoundAssignOperator {{.*}} '>>='
// CHECK:       CompoundAssignOperator {{.*}} '&='
// CHECK:       CompoundAssignOperator {{.*}} '|='
// CHECK:       CompoundAssignOperator {{.*}} '^='
// CHECK:       CompoundAssignOperator {{.*}} '<dependent type>' {{.*}}'+=' ComputeLHSTy='<dependent type>' ComputeResultTy='<dependent type>'
// CHECK-NEXT:    MSPropertySubscriptExpr

#else

void test() {
  V<int> v{{8, 9}};
  v.f(1L);
}

#endif
