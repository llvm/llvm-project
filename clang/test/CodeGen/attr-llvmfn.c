// RUN: %clang_cc1 -debug-info-kind=limited -emit-llvm -o - %s | FileCheck %s

void t1() __attribute__((llvm_fn_attr("custom_attr", "custom_value"), llvm_fn_attr("second_attr", "second_value")));

__attribute__((llvm_fn_attr("third_attr", "third_value")))
void t1()
{
}

// CHECK: define dso_local void @t1() #[[t1attrs:[0-9]+]]

// CHECK: #[[t1attrs]]
// CHECK-SAME: "custom_attr"="custom_value"
// CHECK-SAME: "second_attr"="second_value"
// CHECK-SAME: "third_attr"="third_value"
