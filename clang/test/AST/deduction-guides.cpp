// Test without serialization:
// RUN: %clang_cc1 -fsyntax-only %s -ast-dump -std=c++17 | FileCheck %s
//
// Test with serialization:
// RUN: %clang_cc1 -std=c++17 -emit-pch -o %t %s
// RUN: %clang_cc1 -x c++ -std=c++17 -include-pch %t -ast-dump-all /dev/null \
// RUN: | sed -e "s/ <undeserialized declarations>//" -e "s/ imported//" \
// RUN: | FileCheck %s

namespace PR46111 {
template <typename>
struct S;

template <typename T>
struct HasDeductionGuide {
  typedef PR46111::S<T> STy;
  HasDeductionGuide(typename STy::Child);
};

// This causes deduction guides to be generated for all constructors.
HasDeductionGuide()->HasDeductionGuide<int>;

template <typename T>
struct HasDeductionGuideTypeAlias {
  using STy = PR46111::S<T>;
  HasDeductionGuideTypeAlias(typename STy::Child);
};

// This causes deduction guides to be generated for all constructors.
HasDeductionGuideTypeAlias()->HasDeductionGuideTypeAlias<int>;

// The parameter to this one shouldn't be an elaborated type.
// CHECK: CXXDeductionGuideDecl {{.*}} implicit <deduction guide for HasDeductionGuide> 'auto (typename STy::Child) -> HasDeductionGuide<T>'
// CHECK: CXXDeductionGuideDecl {{.*}} implicit <deduction guide for HasDeductionGuide> 'auto (HasDeductionGuide<T>) -> HasDeductionGuide<T>'
// CHECK: CXXDeductionGuideDecl {{.*}} <deduction guide for HasDeductionGuide> 'auto () -> HasDeductionGuide<int>'
// CHECK: CXXDeductionGuideDecl {{.*}} implicit <deduction guide for HasDeductionGuideTypeAlias> 'auto (typename STy::Child) -> HasDeductionGuideTypeAlias<T>'
// CHECK: CXXDeductionGuideDecl {{.*}} implicit <deduction guide for HasDeductionGuideTypeAlias> 'auto (HasDeductionGuideTypeAlias<T>) -> HasDeductionGuideTypeAlias<T>'
// CHECK: CXXDeductionGuideDecl {{.*}} <deduction guide for HasDeductionGuideTypeAlias> 'auto () -> HasDeductionGuideTypeAlias<int>'
} // namespace PR46111
