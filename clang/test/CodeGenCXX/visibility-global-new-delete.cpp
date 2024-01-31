// RUN: %clang_cc1 %s -std=c++11 -triple x86_64-unknown-unknown -fvisibility=hidden -emit-llvm -o - | FileCheck %s -DLINKAGE=dso_local
// RUN: %clang_cc1 %s -std=c++11 -triple x86_64-unknown-unknown -fvisibility=default -fvisibility-global-new-delete=force-hidden -emit-llvm -o - | FileCheck %s -DLINKAGE=hidden
// RUN: %clang_cc1 %s -std=c++11 -triple x86_64-unknown-unknown -fvisibility=hidden -fvisibility-global-new-delete=force-protected -emit-llvm -o - | FileCheck %s -DLINKAGE=protected
// RUN: %clang_cc1 %s -std=c++11 -triple x86_64-unknown-unknown -fvisibility=hidden -fvisibility-global-new-delete=force-default -emit-llvm -o - | FileCheck %s -DLINKAGE=dso_local
// RUN: %clang_cc1 %s -std=c++11 -triple x86_64-unknown-unknown -fvisibility=hidden -fvisibility-global-new-delete=source -emit-llvm -o - | FileCheck %s -DLINKAGE=hidden

namespace std {
  typedef __typeof__(sizeof(0)) size_t;
  struct nothrow_t {};
}

// Definition which inherits visibility from the implicit compiler generated declaration.
void operator delete(void*) throw() {}
// CHECK: define [[LINKAGE]] void @_ZdlPv
