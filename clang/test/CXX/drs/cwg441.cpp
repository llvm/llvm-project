// RUN: %clang_cc1 -std=c++98 %s -triple x86_64-linux-gnu -emit-llvm -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK
// RUN: %clang_cc1 -std=c++11 %s -triple x86_64-linux-gnu -emit-llvm -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK
// RUN: %clang_cc1 -std=c++14 %s -triple x86_64-linux-gnu -emit-llvm -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK
// RUN: %clang_cc1 -std=c++17 %s -triple x86_64-linux-gnu -emit-llvm -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK
// RUN: %clang_cc1 -std=c++20 %s -triple x86_64-linux-gnu -emit-llvm -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK
// RUN: %clang_cc1 -std=c++23 %s -triple x86_64-linux-gnu -emit-llvm -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK
// RUN: %clang_cc1 -std=c++2c %s -triple x86_64-linux-gnu -emit-llvm -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK

namespace cwg441 { // cwg441: 2.7

struct A {
  A() {}
};

A dynamic_init;
int i;
int& ir = i;
int* ip = &i;

} // namespace cwg441

// CHECK-DAG:   @cwg441::dynamic_init = global %"struct.cwg441::A" zeroinitializer
// CHECK-DAG:   @cwg441::i = global i32 0
// CHECK-DAG:   @cwg441::ir = constant ptr @cwg441::i
// CHECK-DAG:   @cwg441::ip = global ptr @cwg441::i
// CHECK-DAG:   @llvm.global_ctors = appending global [{{.+}}] [{ {{.+}} } { {{.+}}, ptr @_GLOBAL__sub_I_cwg441.cpp, {{.+}} }]

// CHECK-LABEL: define {{.*}} void @__cxx_global_var_init()
// CHECK-NEXT:  entry:
// CHECK-NEXT:    call void @cwg441::A::A()({{.*}} @cwg441::dynamic_init)
// CHECK-NEXT:    ret void
// CHECK-NEXT:  }

// CHECK-LABEL: define {{.*}} void @_GLOBAL__sub_I_cwg441.cpp()
// CHECK-NEXT:  entry:
// CHECK-NEXT:    call void @__cxx_global_var_init()
// CHECK-NEXT:    ret void
// CHECK-NEXT:  }
