//RUN: %clang_cc1 -emit-llvm -Wno-strict-prototypes -triple aarch64-target-linux-gnu %s -o - | FileCheck %s

//CHECK: define dso_local void @test() #0 {
//CHECK: call void @exit(i32 noundef 1)
//CHECK: unreachable

void test(void){
	__builtin_exit(1);
}
