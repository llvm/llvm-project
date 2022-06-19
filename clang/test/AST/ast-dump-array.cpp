// Test without serialization:
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -ast-dump %s \
// RUN: | FileCheck --strict-whitespace %s
//
// Test with serialization:
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -x c++ -triple x86_64-unknown-unknown -include-pch %t -ast-dump-all /dev/null \
// RUN: | sed -e "s/ <undeserialized declarations>//" -e "s/ imported//" \
// RUN: | FileCheck --strict-whitespace %s

void testArrayInitExpr()
{
    int a[10];
    auto l = [a]{
    };
    // CHECK: |-ArrayInitLoopExpr 0x{{[^ ]*}} <col:15> 'int[10]'
    // CHECK: |     `-ArrayInitIndexExpr 0x{{[^ ]*}} <<invalid sloc>> 'unsigned long'
}

template<typename T, int Size>
class array {
  T data[Size];

  using array_T_size = T[Size];
  // CHECK: `-DependentSizedArrayType 0x{{[^ ]*}} 'T[Size]' dependent   <col:25, col:30>
  using const_array_T_size = const T[Size];
  // CHECK: `-DependentSizedArrayType 0x{{[^ ]*}} 'const T[Size]' dependent   <col:37, col:42>
};

struct V {};
template <typename U, typename Idx, int N>
void testDependentSubscript() {
  U* a;
  U b[5];
  Idx i{};
  enum E { One = 1 };

  // Can types of subscript expressions can be determined?
  // LHS is a type-dependent array, RHS is a known integer type.
  a[1];
  // CHECK: ArraySubscriptExpr {{.*}}line:[[@LINE-1]]{{.*}} 'U'
  b[1];
  // CHECK: ArraySubscriptExpr {{.*}}line:[[@LINE-1]]{{.*}} 'U'

  // Reverse case: RHS is a type-dependent array, LHS is an integer.
  1[a];
  // CHECK: ArraySubscriptExpr {{.*}}line:[[@LINE-1]]{{.*}} 'U'
  1[b];
  // CHECK: ArraySubscriptExpr {{.*}}line:[[@LINE-1]]{{.*}} 'U'

  // LHS is a type-dependent array, RHS is type-dependent.
  a[i];
  // CHECK: ArraySubscriptExpr {{.*}}line:[[@LINE-1]]{{.*}} '<dependent type>'
  b[i];
  // CHECK: ArraySubscriptExpr {{.*}}line:[[@LINE-1]]{{.*}} '<dependent type>'

  V *a2;
  V b2[5];

  // LHS is a known array, RHS is type-dependent.
  a2[i];
  // CHECK: ArraySubscriptExpr {{.*}}line:[[@LINE-1]]{{.*}} '<dependent type>'
  b2[i];
  // CHECK: ArraySubscriptExpr {{.*}}line:[[@LINE-1]]{{.*}} '<dependent type>'

  // LHS is a known array, RHS is a type-dependent index.
  // We know the element type is V, but insist on some dependent type.
  a2[One];
  // CHECK: ArraySubscriptExpr {{.*}}line:[[@LINE-1]]{{.*}} '<dependent type>'
  b2[One];
  // CHECK: ArraySubscriptExpr {{.*}}line:[[@LINE-1]]{{.*}} '<dependent type>'

  V b3[N];
  // LHS is an array with dependent bounds but known elements.
  // We insist on a dependent type.
  b3[0];
  // CHECK: ArraySubscriptExpr {{.*}}line:[[@LINE-1]]{{.*}} '<dependent type>'

  U b4[N];
  // LHS is an array with dependent bounds and dependent elements.
  b4[0];
  // CHECK: ArraySubscriptExpr {{.*}}line:[[@LINE-1]]{{.*}} 'U'
}
