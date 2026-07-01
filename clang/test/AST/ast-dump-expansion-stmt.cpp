// Test without serialization:
// RUN: %clang_cc1 -std=c++26 -triple x86_64-unknown-unknown -ast-dump %s
//
// Test with serialization:
// RUN: %clang_cc1 -std=c++26 -triple x86_64-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -x c++ -std=c++26 -triple x86_64-unknown-unknown -include-pch %t -ast-dump-all /dev/null \
// RUN: | sed -e "s/ <undeserialized declarations>//" -e "s/ imported//"

#if 0 // Disabled until we support iterating expansion statements.
template <typename T, __SIZE_TYPE__ size>
struct Array {
  T data[size]{};
  constexpr const T* begin() const { return data; }
  constexpr const T* end() const { return data + size; }
};
#endif // 0

void foo(int);

template <typename T>
void test(T t) {
  // CHECK:      CXXExpansionStmtDecl
  // CHECK-NEXT:   CXXExpansionStmtPattern {{.*}} enumerating
  // CHECK:        CXXExpansionStmtInstantiation
  template for (auto x : {1, 2, 3}) {
    foo(x);
  }

#if 0 // Disabled until we support iterating expansion statements.
  // NOTE: Remove 'DISABLED-' when the '#if 0' is removed.
  // DISABLED-CHECK:      CXXExpansionStmtDecl
  // DISABLED-CHECK-NEXT:   CXXExpansionStmtPattern {{.*}} iterating
  // DISABLED-CHECK:        CXXExpansionStmtInstantiation
  static constexpr Array<int, 3> a;
  template for (auto x : a) {
    foo(x);
  }
#endif

  // CHECK:      CXXExpansionStmtDecl
  // CHECK-NEXT:   CXXExpansionStmtPattern {{.*}} destructuring
  // CHECK:        CXXExpansionStmtInstantiation
  int arr[3]{1, 2, 3};
  template for (auto x : arr) {
    foo(x);
  }

  // CHECK:      CXXExpansionStmtDecl
  // CHECK-NEXT:   CXXExpansionStmtPattern {{.*}} dependent
  // CHECK-NOT:    CXXExpansionStmtInstantiation
  template for (auto x : t) {
    foo(x);
  }
}
