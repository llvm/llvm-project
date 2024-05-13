// RUN: %clang_cc1 -std=c++20 -ast-dump %s -ast-dump-filter Test | FileCheck %s

namespace std {
  struct strong_ordering {
    int n;
    constexpr operator int() const { return n; }
    static const strong_ordering less, equal, greater;
  };
  constexpr strong_ordering strong_ordering::less{-1},
      strong_ordering::equal{0}, strong_ordering::greater{1};
}

template <typename T, typename U>
auto Test(T* pt, U* pu) {
  // CHECK: BinaryOperator {{.*}} '<dependent type>' '<=>'
  // CHECK-NEXT: DeclRefExpr {{.*}} 'T *' lvalue ParmVar {{.*}} 'pt' 'T *'
  // CHECK-NEXT: DeclRefExpr {{.*}} 'U *' lvalue ParmVar {{.*}} 'pu' 'U *'
  (void)(pt <=> pu);

}


