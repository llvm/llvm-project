// RUN: %clang_cc1 -debug-info-kind=limited -emit-llvm -o - | FileCheck %s

void t1() __attribute__((llvm_fn_attr("custom_attr", "custom_value"), llvm_fn_attr("second_attr", "second_value")));

void t1()
{
}

void t2();

void t3() {
	t2() ____attribute__((llvm_fn_attr("custom_attr", "custom_value"), llvm_fn_attr("second_attr", "second_value")));
}

