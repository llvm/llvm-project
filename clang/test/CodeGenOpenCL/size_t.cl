// RUN: %clang_cc1 %s -cl-std=CL2.0 -finclude-default-header -emit-llvm -O0 -triple spir-unknown-unknown -o - | FileCheck --check-prefix=SZ32 %s
// RUN: %clang_cc1 %s -cl-std=CL2.0 -finclude-default-header -emit-llvm -O0 -triple spir64-unknown-unknown -o - | FileCheck --check-prefix=SZ64 --check-prefix=SZ64ONLY %s
// RUN: %clang_cc1 %s -cl-std=CL2.0 -finclude-default-header -emit-llvm -O0 -triple amdgcn -o - | FileCheck --check-prefix=SZ64 --check-prefix=AMDONLY %s
// RUN: %clang_cc1 %s -cl-std=CL2.0 -finclude-default-header -emit-llvm -O0 -triple amdgcn---opencl -o - | FileCheck --check-prefix=SZ64 --check-prefix=AMDONLY %s

//SZ32: define{{.*}} i32 @test_ptrtoint_private(i8* %x)
//SZ32: ptrtoint i8* %{{.*}} to i32
//SZ64: define{{.*}} i64 @test_ptrtoint_private(i8* %x)
//SZ64: ptrtoint i8* %{{.*}} to i64
size_t test_ptrtoint_private(private char* x) {
  return (size_t)x;
}

//SZ32: define{{.*}} i32 @test_ptrtoint_global(i8 addrspace(1)* %x)
//SZ32: ptrtoint i8 addrspace(1)* %{{.*}} to i32
//SZ64: define{{.*}} i64 @test_ptrtoint_global(i8 addrspace(1)* %x)
//SZ64: ptrtoint i8 addrspace(1)* %{{.*}} to i64
intptr_t test_ptrtoint_global(global char* x) {
  return (intptr_t)x;
}

//SZ32: define{{.*}} i32 @test_ptrtoint_constant(i8 addrspace(2)* %x)
//SZ32: ptrtoint i8 addrspace(2)* %{{.*}} to i32
//SZ64: define{{.*}} i64 @test_ptrtoint_constant(i8 addrspace(2)* %x)
//SZ64: ptrtoint i8 addrspace(2)* %{{.*}} to i64
uintptr_t test_ptrtoint_constant(constant char* x) {
  return (uintptr_t)x;
}

//SZ32: define{{.*}} i32 @test_ptrtoint_local(i8 addrspace(3)* %x)
//SZ32: ptrtoint i8 addrspace(3)* %{{.*}} to i32
//SZ64: define{{.*}} i64 @test_ptrtoint_local(i8 addrspace(3)* %x)
//SZ64: ptrtoint i8 addrspace(3)* %{{.*}} to i64
size_t test_ptrtoint_local(local char* x) {
  return (size_t)x;
}

//SZ32: define{{.*}} i32 @test_ptrtoint_generic(i8 addrspace(4)* %x)
//SZ32: ptrtoint i8 addrspace(4)* %{{.*}} to i32
//SZ64: define{{.*}} i64 @test_ptrtoint_generic(i8 addrspace(4)* %x)
//SZ64: ptrtoint i8 addrspace(4)* %{{.*}} to i64
size_t test_ptrtoint_generic(generic char* x) {
  return (size_t)x;
}

//SZ32: define{{.*}} i8* @test_inttoptr_private(i32 %x)
//SZ32: inttoptr i32 %{{.*}} to i8*
//SZ64: define{{.*}} i8* @test_inttoptr_private(i64 %x)
//AMDONLY: trunc i64 %{{.*}} to i32
//AMDONLY: inttoptr i32 %{{.*}} to i8*
//SZ64ONLY: inttoptr i64 %{{.*}} to i8*
private char* test_inttoptr_private(size_t x) {
  return (private char*)x;
}

//SZ32: define{{.*}} i8 addrspace(1)* @test_inttoptr_global(i32 %x)
//SZ32: inttoptr i32 %{{.*}} to i8 addrspace(1)*
//SZ64: define{{.*}} i8 addrspace(1)* @test_inttoptr_global(i64 %x)
//SZ64: inttoptr i64 %{{.*}} to i8 addrspace(1)*
global char* test_inttoptr_global(size_t x) {
  return (global char*)x;
}

//SZ32: define{{.*}} i8 addrspace(3)* @test_add_local(i8 addrspace(3)* %x, i32 %y)
//SZ32: getelementptr inbounds i8, i8 addrspace(3)* %{{.*}}, i32
//SZ64: define{{.*}} i8 addrspace(3)* @test_add_local(i8 addrspace(3)* %x, i64 %y)
//AMDONLY: trunc i64 %{{.*}} to i32
//AMDONLY: getelementptr inbounds i8, i8 addrspace(3)* %{{.*}}, i32
//SZ64ONLY: getelementptr inbounds i8, i8 addrspace(3)* %{{.*}}, i64
local char* test_add_local(local char* x, ptrdiff_t y) {
  return x + y;
}

//SZ32: define{{.*}} i8 addrspace(1)* @test_add_global(i8 addrspace(1)* %x, i32 %y)
//SZ32: getelementptr inbounds i8, i8 addrspace(1)* %{{.*}}, i32
//SZ64: define{{.*}} i8 addrspace(1)* @test_add_global(i8 addrspace(1)* %x, i64 %y)
//SZ64: getelementptr inbounds i8, i8 addrspace(1)* %{{.*}}, i64
global char* test_add_global(global char* x, ptrdiff_t y) {
  return x + y;
}

//SZ32: define{{.*}} i32 @test_sub_local(i8 addrspace(3)* %x, i8 addrspace(3)* %y)
//SZ32: ptrtoint i8 addrspace(3)* %{{.*}} to i32
//SZ32: ptrtoint i8 addrspace(3)* %{{.*}} to i32
//SZ64: define{{.*}} i64 @test_sub_local(i8 addrspace(3)* %x, i8 addrspace(3)* %y)
//SZ64: ptrtoint i8 addrspace(3)* %{{.*}} to i64
//SZ64: ptrtoint i8 addrspace(3)* %{{.*}} to i64
ptrdiff_t test_sub_local(local char* x, local char *y) {
  return x - y;
}

//SZ32: define{{.*}} i32 @test_sub_private(i8* %x, i8* %y)
//SZ32: ptrtoint i8* %{{.*}} to i32
//SZ32: ptrtoint i8* %{{.*}} to i32
//SZ64: define{{.*}} i64 @test_sub_private(i8* %x, i8* %y)
//SZ64: ptrtoint i8* %{{.*}} to i64
//SZ64: ptrtoint i8* %{{.*}} to i64
ptrdiff_t test_sub_private(private char* x, private char *y) {
  return x - y;
}

//SZ32: define{{.*}} i32 @test_sub_mix(i8* %x, i8 addrspace(4)* %y)
//SZ32: ptrtoint i8* %{{.*}} to i32
//SZ32: ptrtoint i8 addrspace(4)* %{{.*}} to i32
//SZ64: define{{.*}} i64 @test_sub_mix(i8* %x, i8 addrspace(4)* %y)
//SZ64: ptrtoint i8* %{{.*}} to i64
//SZ64: ptrtoint i8 addrspace(4)* %{{.*}} to i64
ptrdiff_t test_sub_mix(private char* x, generic char *y) {
  return x - y;
}

