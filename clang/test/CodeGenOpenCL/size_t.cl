// RUN: %clang_cc1 %s -cl-std=CL2.0 -finclude-default-header -fdeclare-opencl-builtins -emit-llvm -O0 -triple spir-unknown-unknown -o - | FileCheck --check-prefix=SZ32 %s
// RUN: %clang_cc1 %s -cl-std=CL2.0 -finclude-default-header -fdeclare-opencl-builtins -emit-llvm -O0 -triple spir64-unknown-unknown -o - | FileCheck --check-prefix=SZ64 --check-prefix=SZ64ONLY %s
// RUN: %clang_cc1 %s -cl-std=CL2.0 -finclude-default-header -fdeclare-opencl-builtins -emit-llvm -O0 -triple amdgcn -o - | FileCheck --check-prefix=SZ64 --check-prefix=AMDGCN %s
// RUN: %clang_cc1 %s -cl-std=CL2.0 -finclude-default-header -fdeclare-opencl-builtins -emit-llvm -O0 -triple amdgcn---opencl -o - | FileCheck --check-prefix=SZ64 --check-prefix=AMDGCN %s

//SZ32: define{{.*}} i32 @test_ptrtoint_private(ptr noundef %x)
//SZ32: ptrtoint ptr %{{.*}} to i32
//SZ64ONLY: define{{.*}} i64 @test_ptrtoint_private(ptr noundef %x)
//SZ64ONLY: ptrtoint ptr %{{.*}} to i64
//AMDGCN: define{{.*}} i64 @test_ptrtoint_private(ptr addrspace(5) noundef %x)
//AMDGCN: ptrtoint ptr addrspace(5) %{{.*}} to i64
size_t test_ptrtoint_private(private char* x) {
  return (size_t)x;
}

//SZ32: define{{.*}} i32 @test_ptrtoint_global(ptr addrspace(1) noundef %x)
//SZ32: ptrtoint ptr addrspace(1) %{{.*}} to i32
//SZ64: define{{.*}} i64 @test_ptrtoint_global(ptr addrspace(1) noundef %x)
//SZ64: ptrtoint ptr addrspace(1) %{{.*}} to i64
intptr_t test_ptrtoint_global(global char* x) {
  return (intptr_t)x;
}

//SZ32: define{{.*}} i32 @test_ptrtoint_constant(ptr addrspace(2) noundef %x)
//SZ32: ptrtoint ptr addrspace(2) %{{.*}} to i32
//SZ64ONLY: define{{.*}} i64 @test_ptrtoint_constant(ptr addrspace(2) noundef %x)
//SZ64ONLY: ptrtoint ptr addrspace(2) %{{.*}} to i64
//AMDGCN: define{{.*}} i64 @test_ptrtoint_constant(ptr addrspace(4) noundef %x)
//AMDGCN: ptrtoint ptr addrspace(4) %{{.*}} to i64
uintptr_t test_ptrtoint_constant(constant char* x) {
  return (uintptr_t)x;
}

//SZ32: define{{.*}} i32 @test_ptrtoint_local(ptr addrspace(3) noundef %x)
//SZ32: ptrtoint ptr addrspace(3) %{{.*}} to i32
//SZ64: define{{.*}} i64 @test_ptrtoint_local(ptr addrspace(3) noundef %x)
//SZ64: ptrtoint ptr addrspace(3) %{{.*}} to i64
size_t test_ptrtoint_local(local char* x) {
  return (size_t)x;
}

//SZ32: define{{.*}} i32 @test_ptrtoint_generic(ptr addrspace(4) noundef %x)
//SZ32: ptrtoint ptr addrspace(4) %{{.*}} to i32
//SZ64ONLY: define{{.*}} i64 @test_ptrtoint_generic(ptr addrspace(4) noundef %x)
//SZ64ONLY: ptrtoint ptr addrspace(4) %{{.*}} to i64
//AMDGCN: define{{.*}} i64 @test_ptrtoint_generic(ptr noundef %x)
//AMDGCN: ptrtoint ptr %{{.*}} to i64
size_t test_ptrtoint_generic(generic char* x) {
  return (size_t)x;
}

//SZ32: define{{.*}} ptr @test_inttoptr_private(i32 noundef %x)
//SZ32: inttoptr i32 %{{.*}} to ptr
//SZ64ONLY: define{{.*}} ptr @test_inttoptr_private(i64 noundef %x)
//SZ64ONLY: inttoptr i64 %{{.*}} to ptr
//AMDGCN: define{{.*}} ptr addrspace(5) @test_inttoptr_private(i64 noundef %x)
//AMDGCN: trunc i64 %{{.*}} to i32
//AMDGCN: inttoptr i32 %{{.*}} to ptr addrspace(5)
private char* test_inttoptr_private(size_t x) {
  return (private char*)x;
}

//SZ32: define{{.*}} ptr addrspace(1) @test_inttoptr_global(i32 noundef %x)
//SZ32: inttoptr i32 %{{.*}} to ptr addrspace(1)
//SZ64: define{{.*}} ptr addrspace(1) @test_inttoptr_global(i64 noundef %x)
//SZ64: inttoptr i64 %{{.*}} to ptr addrspace(1)
global char* test_inttoptr_global(size_t x) {
  return (global char*)x;
}

//SZ32: define{{.*}} ptr addrspace(3) @test_add_local(ptr addrspace(3) noundef %x, i32 noundef %y)
//SZ32: getelementptr inbounds i8, ptr addrspace(3) %{{.*}}, i32
//SZ64: define{{.*}} ptr addrspace(3) @test_add_local(ptr addrspace(3) noundef %x, i64 noundef %y)
//AMDGCN: trunc i64 %{{.*}} to i32
//AMDGCN: getelementptr inbounds i8, ptr addrspace(3) %{{.*}}, i32
//SZ64ONLY: getelementptr inbounds i8, ptr addrspace(3) %{{.*}}, i64
local char* test_add_local(local char* x, ptrdiff_t y) {
  return x + y;
}

//SZ32: define{{.*}} ptr addrspace(1) @test_add_global(ptr addrspace(1) noundef %x, i32 noundef %y)
//SZ32: getelementptr inbounds i8, ptr addrspace(1) %{{.*}}, i32
//SZ64: define{{.*}} ptr addrspace(1) @test_add_global(ptr addrspace(1) noundef %x, i64 noundef %y)
//SZ64: getelementptr inbounds i8, ptr addrspace(1) %{{.*}}, i64
global char* test_add_global(global char* x, ptrdiff_t y) {
  return x + y;
}

//SZ32: define{{.*}} i32 @test_sub_local(ptr addrspace(3) noundef %x, ptr addrspace(3) noundef %y)
//SZ32: ptrtoint ptr addrspace(3) %{{.*}} to i32
//SZ32: ptrtoint ptr addrspace(3) %{{.*}} to i32
//SZ64: define{{.*}} i64 @test_sub_local(ptr addrspace(3) noundef %x, ptr addrspace(3) noundef %y)
//SZ64: ptrtoint ptr addrspace(3) %{{.*}} to i64
//SZ64: ptrtoint ptr addrspace(3) %{{.*}} to i64
ptrdiff_t test_sub_local(local char* x, local char *y) {
  return x - y;
}

//SZ32: define{{.*}} i32 @test_sub_private(ptr noundef %x, ptr noundef %y)
//SZ32: ptrtoint ptr %{{.*}} to i32
//SZ32: ptrtoint ptr %{{.*}} to i32
//SZ64ONLY: define{{.*}} i64 @test_sub_private(ptr noundef %x, ptr noundef %y)
//SZ64ONLY: ptrtoint ptr %{{.*}} to i64
//SZ64ONLY: ptrtoint ptr %{{.*}} to i64
//AMDGCN: define{{.*}} i64 @test_sub_private(ptr addrspace(5) noundef %x, ptr addrspace(5) noundef %y)
//AMDGCN: ptrtoint ptr addrspace(5) %{{.*}} to i64
//AMDGCN: ptrtoint ptr addrspace(5) %{{.*}} to i64
ptrdiff_t test_sub_private(private char* x, private char *y) {
  return x - y;
}

//SZ32: define{{.*}} i32 @test_sub_mix(ptr noundef %x, ptr addrspace(4) noundef %y)
//SZ32: ptrtoint ptr %{{.*}} to i32
//SZ32: ptrtoint ptr addrspace(4) %{{.*}} to i32
//SZ64ONLY: define{{.*}} i64 @test_sub_mix(ptr noundef %x, ptr addrspace(4) noundef %y)
//SZ64ONLY: ptrtoint ptr %{{.*}} to i64
//SZ64ONLY: ptrtoint ptr addrspace(4) %{{.*}} to i64
//AMDGCN: define{{.*}} i64 @test_sub_mix(ptr addrspace(5) noundef %x, ptr noundef %y)
//AMDGCN: ptrtoint ptr addrspace(5) %{{.*}} to i64
//AMDGCN: ptrtoint ptr %{{.*}} to i64
ptrdiff_t test_sub_mix(private char* x, generic char *y) {
  return x - y;
}

