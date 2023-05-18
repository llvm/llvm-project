// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -o - %s -std=c++14 | FileCheck %s

template<typename> struct custom_copy_ctor {
  custom_copy_ctor() = default;
  custom_copy_ctor(custom_copy_ctor const &) {}
};

// CHECK: define {{.*}} @_ZN16custom_copy_ctorIvEC2ERKS0_(
void pr22354() {
  custom_copy_ctor<void> cc;
  [cc](auto){}(1);
}

