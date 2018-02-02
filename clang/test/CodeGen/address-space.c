// RUN: %clang_cc1 -triple x86_64-apple-darwin -emit-llvm < %s | FileCheck -check-prefixes=CHECK,X86,GIZ %s
// RUN: %clang_cc1 -triple amdgcn---amdgiz -emit-llvm < %s | FileCheck -check-prefixes=CHECK,AMDGIZ,GIZ %s

// CHECK: @foo = common addrspace(1) global
int foo __attribute__((address_space(1)));

// CHECK: @ban = common addrspace(1) global
int ban[10] __attribute__((address_space(1)));

// CHECK: @a = common global
int a __attribute__((address_space(0)));

// CHECK-LABEL: define i32 @test1() 
// CHECK: load i32, i32 addrspace(1)* @foo
int test1() { return foo; }

// CHECK-LABEL: define i32 @test2(i32 %i) 
// CHECK: load i32, i32 addrspace(1)*
// CHECK-NEXT: ret i32
int test2(int i) { return ban[i]; }

// Both A and B point into addrspace(2).
__attribute__((address_space(2))) int *A, *B;

// CHECK-LABEL: define void @test3()
// X86: load i32 addrspace(2)*, i32 addrspace(2)** @B
// AMDGIZ: load i32 addrspace(2)*, i32 addrspace(2)** addrspacecast (i32 addrspace(2)* addrspace(1)* @B to i32 addrspace(2)**)
// CHECK: load i32, i32 addrspace(2)*
// X86: load i32 addrspace(2)*, i32 addrspace(2)** @A
// AMDGIZ: load i32 addrspace(2)*, i32 addrspace(2)** addrspacecast (i32 addrspace(2)* addrspace(1)* @A to i32 addrspace(2)**)
// CHECK: store i32 {{.*}}, i32 addrspace(2)*
void test3() {
  *A = *B;
}

// PR7437
typedef struct {
  float aData[1];
} MyStruct;

// CHECK-LABEL: define void @test4(
// GIZ: call void @llvm.memcpy.p0i8.p2i8
// GIZ: call void @llvm.memcpy.p2i8.p0i8
void test4(MyStruct __attribute__((address_space(2))) *pPtr) {
  MyStruct s = pPtr[0];
  pPtr[0] = s;
}
