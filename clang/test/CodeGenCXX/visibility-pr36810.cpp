// RUN: %clang_cc1 -triple x86_64-apple-macosx -std=c++11 -fvisibility hidden -emit-llvm -o - %s -O2 -disable-llvm-passes | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-apple-macosx -DUNDEF_G -std=c++11 -fvisibility hidden -emit-llvm -o - %s -O2 -disable-llvm-passes | FileCheck %s

namespace std {
template <class>
class __attribute__((__type_visibility__("default"))) shared_ptr {
  template <class> friend class shared_ptr;
};
}
struct dict;
#ifndef UNDEF_G
std::shared_ptr<dict> g;
#endif
class __attribute__((visibility("default"))) Bar;
template <class = std::shared_ptr<Bar>>
class __attribute__((visibility("default"))) i {
  std::shared_ptr<int> foo() const;
};

// CHECK: define{{.*}} void @_ZNK1iISt10shared_ptrI3BarEE3fooEv
template <> std::shared_ptr<int> i<>::foo() const {
  return std::shared_ptr<int>();
}
