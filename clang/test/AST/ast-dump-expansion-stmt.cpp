// Test without serialization:
// RUN: %clang_cc1 -std=c++26 -triple x86_64-unknown-unknown -ast-dump %s
//
// Test with serialization:
// RUN: %clang_cc1 -std=c++26 -triple x86_64-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -x c++ -std=c++26 -triple x86_64-unknown-unknown -include-pch %t -ast-dump-all /dev/null \
// RUN: | sed -e "s/ <undeserialized declarations>//" -e "s/ imported//"

template <typename T, __SIZE_TYPE__ size>
struct Array {
  T data[size]{};
  constexpr const T* begin() const { return data; }
  constexpr const T* end() const { return data + size; }
};

void foo(int);

template <typename T>
void test(T t) {
  // CHECK:      ExpansionStmtDecl
  // CHECK-NEXT:   CXXEnumeratingExpansionStmt
  // CHECK:        CXXExpansionInstantiationStmt
  template for (auto x : {1, 2, 3}) {
    foo(x);
  }

  // CHECK:      ExpansionStmtDecl
  // CHECK-NEXT:   CXXIteratingExpansionStmt
  // CHECK:        CXXExpansionInstantiationStmt
  static constexpr Array<int, 3> a;
  template for (auto x : a) {
    foo(x);
  }

  // CHECK:      ExpansionStmtDecl
  // CHECK-NEXT:   CXXDestructuringExpansionStmt
  // CHECK:        CXXExpansionInstantiationStmt
  int arr[3]{1, 2, 3};
  template for (auto x : arr) {
    foo(x);
  }

  // CHECK:      ExpansionStmtDecl
  // CHECK-NEXT:   CXXDependentExpansionStmt
  // CHECK-NOT:    CXXExpansionInstantiationStmt
  template for (auto x : t) {
    foo(x);
  }
}
