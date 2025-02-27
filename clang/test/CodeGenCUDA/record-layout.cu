// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -fdump-record-layouts \
// RUN:   -emit-llvm -o %t -xhip %s 2>&1 | FileCheck %s --check-prefix=AST
// RUN: cat %t | FileCheck --check-prefixes=CHECK,HOST %s
// RUN: %clang_cc1 -fcuda-is-device -triple amdgcn-amd-amdhsa -target-cpu gfx1100 \
// RUN:   -emit-llvm -fdump-record-layouts -aux-triple x86_64-pc-windows-msvc \
// RUN:   -o %t -xhip %s | FileCheck %s --check-prefix=AST
// RUN: cat %t | FileCheck --check-prefixes=CHECK,DEV %s

#include "Inputs/cuda.h"

// AST: *** Dumping AST Record Layout
// AST-LABEL:         0 | struct C
// AST-NEXT:          0 |   struct A (base) (empty)
// AST-NEXT:          1 |   struct B (base) (empty)
// AST-NEXT:          4 |   int i
// AST-NEXT:            | [sizeof=8, align=4,
// AST-NEXT:            |  nvsize=8, nvalign=4]

// CHECK: %struct.C = type { [4 x i8], i32 }

struct A {};
struct B {};
struct C : A, B {
    int i;
};

// AST: *** Dumping AST Record Layout
// AST-LABEL:          0 | struct I
// AST-NEXT:           0 |   (I vftable pointer)
// AST-NEXT:           8 |   int i
// AST-NEXT:             | [sizeof=16, align=8,
// AST-NEXT:             |  nvsize=16, nvalign=8]

// AST: *** Dumping AST Record Layout
// AST-LABEL:          0 | struct J
// AST-NEXT:           0 |   struct I (primary base)
// AST-NEXT:           0 |     (I vftable pointer)
// AST-NEXT:           8 |     int i
// AST-NEXT:          16 |   int j
// AST-NEXT:             | [sizeof=24, align=8,
// AST-NEXT:             |  nvsize=24, nvalign=8]

// CHECK: %struct.I = type { ptr, i32 }
// CHECK: %struct.J = type { %struct.I, i32 }

// HOST: @0 = private unnamed_addr constant { [4 x ptr] } { [4 x ptr] [ptr @"??_R4J@@6B@", ptr @"?f@J@@UEAAXXZ", ptr null, ptr @"?h@J@@UEAAXXZ"] }, comdat($"??_7J@@6B@")
// HOST: @1 = private unnamed_addr constant { [4 x ptr] } { [4 x ptr] [ptr @"??_R4I@@6B@", ptr @_purecall, ptr null, ptr @_purecall] }, comdat($"??_7I@@6B@")
// HOST: @"??_7J@@6B@" = unnamed_addr alias ptr, getelementptr inbounds ({ [4 x ptr] }, ptr @0, i32 0, i32 0, i32 1)
// HOST: @"??_7I@@6B@" = unnamed_addr alias ptr, getelementptr inbounds ({ [4 x ptr] }, ptr @1, i32 0, i32 0, i32 1)

// DEV: @_ZTV1J = linkonce_odr unnamed_addr addrspace(1) constant { [5 x ptr addrspace(1)] } { [5 x ptr addrspace(1)] [ptr addrspace(1) null, ptr addrspace(1) null, ptr addrspace(1) null, ptr addrspace(1) addrspacecast (ptr @_ZN1J1gEv to ptr addrspace(1)), ptr addrspace(1) addrspacecast (ptr @_ZN1J1hEv to ptr addrspace(1))] }, comdat, align 8
// DEV: @_ZTV1I = linkonce_odr unnamed_addr addrspace(1) constant { [5 x ptr addrspace(1)] } { [5 x ptr addrspace(1)] [ptr addrspace(1) null, ptr addrspace(1) null, ptr addrspace(1) null, ptr addrspace(1) addrspacecast (ptr @__cxa_pure_virtual to ptr addrspace(1)), ptr addrspace(1) addrspacecast (ptr @__cxa_pure_virtual to ptr addrspace(1))] }, comdat, align 8
struct I {
    virtual void f() = 0;
    __device__ virtual void g() = 0;
    __device__ __host__ virtual void h() = 0;
    int i;
};

struct J : I {
    void f() override {}
    __device__ void g() override {}
    __device__ __host__ void h() override {}
    int j;
};

// DEV: define dso_local amdgpu_kernel void @_Z8C_kernel1C(ptr addrspace(4) noundef byref(%struct.C) align 4 %0)
// DEV:  %coerce = alloca %struct.C, align 4, addrspace(5)
// DEV:  %c = addrspacecast ptr addrspace(5) %coerce to ptr
// DEV:  call void @llvm.memcpy.p0.p4.i64(ptr align 4 %c, ptr addrspace(4) align 4 %0, i64 8, i1 false)
// DEV:  %i = getelementptr inbounds nuw %struct.C, ptr %c, i32 0, i32 1
// DEV:  store i32 1, ptr %i, align 4

__global__ void C_kernel(C c)
{
  c.i = 1;
}

// HOST-LABEL: define dso_local void @"?test_C@@YAXXZ"()
// HOST:  %c = alloca %struct.C, align 4
// HOST:  %i = getelementptr inbounds nuw %struct.C, ptr %c, i32 0, i32 1
// HOST:  store i32 11, ptr %i, align 4

void test_C() {
  C c;
  c.i = 11;
  C_kernel<<<1, 1>>>(c);
}

// DEV: define dso_local void @_Z5J_devP1J(ptr noundef %j)
// DEV:  %j.addr = alloca ptr, align 8, addrspace(5)
// DEV:  %j.addr.ascast = addrspacecast ptr addrspace(5) %j.addr to ptr
// DEV:  store ptr %j, ptr %j.addr.ascast, align 8
// DEV:  %0 = load ptr, ptr %j.addr.ascast, align 8
// DEV:  %i = getelementptr inbounds nuw %struct.I, ptr %0, i32 0, i32 1
// DEV:  store i32 2, ptr %i, align 8
// DEV:  %1 = load ptr, ptr %j.addr.ascast, align 8
// DEV:  %j1 = getelementptr inbounds nuw %struct.J, ptr %1, i32 0, i32 1
// DEV:  store i32 3, ptr %j1, align 8
// DEV:  %2 = load ptr, ptr %j.addr.ascast, align 8
// DEV:  %vtable = load ptr addrspace(1), ptr %2, align 8
// DEV:  %vfn = getelementptr inbounds ptr addrspace(1), ptr addrspace(1) %vtable, i64 1
// DEV:  %3 = load ptr addrspace(1), ptr addrspace(1) %vfn, align 8
// DEV:  call addrspace(1) void %3(ptr noundef nonnull align 8 dereferenceable(24) %2)
// DEV:  %4 = load ptr, ptr %j.addr.ascast, align 8
// DEV:  %vtable2 = load ptr addrspace(1), ptr %4, align 8
// DEV:  %vfn3 = getelementptr inbounds ptr addrspace(1), ptr addrspace(1) %vtable2, i64 2
// DEV:  %5 = load ptr addrspace(1), ptr addrspace(1) %vfn3, align 8
// DEV:  call addrspace(1) void %5(ptr noundef nonnull align 8 dereferenceable(24) %4)

__device__ void J_dev(J *j) {
  j->i = 2;
  j->j = 3;
  j->g();
  j->h();
}

// DEV: define dso_local amdgpu_kernel void @_Z8J_kernelv()
// DEV:  %j = alloca %struct.J, align 8, addrspace(5)
// DEV:  %j.ascast = addrspacecast ptr addrspace(5) %j to ptr
// DEV:  call void @_ZN1JC1Ev(ptr noundef nonnull align 8 dereferenceable(24) %j.ascast)
// DEV:  call void @_Z5J_devP1J(ptr noundef %j.ascast)

__global__ void J_kernel() {
  J j;
  J_dev(&j);
}

// HOST-LABEL: define dso_local void @"?J_host@@YAXPEAUJ@@@Z"(ptr noundef %j)
// HOST:  %0 = load ptr, ptr %j.addr, align 8
// HOST:  %i = getelementptr inbounds nuw %struct.I, ptr %0, i32 0, i32 1
// HOST:  store i32 12, ptr %i, align 8
// HOST:  %1 = load ptr, ptr %j.addr, align 8
// HOST:  %j1 = getelementptr inbounds nuw %struct.J, ptr %1, i32 0, i32 1
// HOST:  store i32 13, ptr %j1, align 8
// HOST:  %2 = load ptr, ptr %j.addr, align 8
// HOST:  %vtable = load ptr, ptr %2, align 8
// HOST:  %vfn = getelementptr inbounds ptr, ptr %vtable, i64 0
// HOST:  %3 = load ptr, ptr %vfn, align 8
// HOST:  call void %3(ptr noundef nonnull align 8 dereferenceable(24) %2)
// HOST:  %4 = load ptr, ptr %j.addr, align 8
// HOST:  %vtable2 = load ptr, ptr %4, align 8
// HOST:  %vfn3 = getelementptr inbounds ptr, ptr %vtable2, i64 2
// HOST:  %5 = load ptr, ptr %vfn3, align 8
// HOST:  call void %5(ptr noundef nonnull align 8 dereferenceable(24) %4)

void J_host(J *j) {
  j->i = 12;
  j->j = 13;
  j->f();
  j->h();
}

// HOST: define dso_local void @"?test_J@@YAXXZ"()
// HOST:  %j = alloca %struct.J, align 8
// HOST:  %call = call noundef ptr @"??0J@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(24) %j)
// HOST:  call void @"?J_host@@YAXPEAUJ@@@Z"(ptr noundef %j)

void test_J() {
  J j;
  J_host(&j);
  J_kernel<<<1, 1>>>();
}

// HOST: define linkonce_odr dso_local noundef ptr @"??0J@@QEAA@XZ"(ptr noundef nonnull returned align 8 dereferenceable(24) %this)
// HOST:  %this.addr = alloca ptr, align 8
// HOST:  store ptr %this, ptr %this.addr, align 8
// HOST:  %this1 = load ptr, ptr %this.addr, align 8
// HOST:  %call = call noundef ptr @"??0I@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(16) %this1) #5
// HOST:  store ptr @"??_7J@@6B@", ptr %this1, align 8
// HOST:  ret ptr %this1

// HOST: define linkonce_odr dso_local noundef ptr @"??0I@@QEAA@XZ"(ptr noundef nonnull returned align 8 dereferenceable(16) %this)
// HOST:  %this.addr = alloca ptr, align 8
// HOST:  store ptr %this, ptr %this.addr, align 8
// HOST:  %this1 = load ptr, ptr %this.addr, align 8
// HOST:  store ptr @"??_7I@@6B@", ptr %this1, align 8
// HOST:  ret ptr %this1

// DEV: define linkonce_odr void @_ZN1JC1Ev(ptr noundef nonnull align 8 dereferenceable(24) %this)
// DEV:  %this.addr = alloca ptr, align 8, addrspace(5)
// DEV:  %this.addr.ascast = addrspacecast ptr addrspace(5) %this.addr to ptr
// DEV:  store ptr %this, ptr %this.addr.ascast, align 8
// DEV:  %this1 = load ptr, ptr %this.addr.ascast, align 8
// DEV:  call void @_ZN1JC2Ev(ptr noundef nonnull align 8 dereferenceable(24) %this1)

// DEV: define linkonce_odr void @_ZN1JC2Ev(ptr noundef nonnull align 8 dereferenceable(24) %this)
// DEV:  %this.addr = alloca ptr, align 8, addrspace(5)
// DEV:  %this.addr.ascast = addrspacecast ptr addrspace(5) %this.addr to ptr
// DEV:  store ptr %this, ptr %this.addr.ascast, align 8
// DEV:  %this1 = load ptr, ptr %this.addr.ascast, align 8
// DEV:  call void @_ZN1IC2Ev(ptr noundef nonnull align 8 dereferenceable(16) %this1)
// DEV:  store ptr addrspace(1) getelementptr inbounds inrange(-16, 24) ({ [5 x ptr addrspace(1)] }, ptr addrspace(1) @_ZTV1J, i32 0, i32 0, i32 2), ptr %this1, align 8

// DEV: define linkonce_odr void @_ZN1IC2Ev(ptr noundef nonnull align 8 dereferenceable(16) %this)
// DEV:  %this.addr = alloca ptr, align 8, addrspace(5)
// DEV:  %this.addr.ascast = addrspacecast ptr addrspace(5) %this.addr to ptr
// DEV:  store ptr %this, ptr %this.addr.ascast, align 8
// DEV:  %this1 = load ptr, ptr %this.addr.ascast, align 8
// DEV:  store ptr addrspace(1) getelementptr inbounds inrange(-16, 24) ({ [5 x ptr addrspace(1)] }, ptr addrspace(1) @_ZTV1I, i32 0, i32 0, i32 2), ptr %this1, align 8
