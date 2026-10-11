// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

// FIXME: Add LLVM and OGCG checks for the virtual base once cir.vtable.get_vptr
// keeps the address space.
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -DVBASE %s -o %t-vbase.cir
// RUN: FileCheck --input-file=%t-vbase.cir %s -check-prefix=CIR-VBASE

// A base or derived class subobject is in the address space of the object.

#define AS1 __attribute__((address_space(1)))

struct A { int a; };
struct B { int b; };
struct D : A, B { int d; };

int base_member(AS1 D *d) { return d->a; }

// CIR-LABEL: cir.func {{.*}}@_Z11base_memberPU3AS11D
// CIR: cir.base_class_addr nonnull %{{.*}} [0] : !cir.ptr<!rec_D, target_address_space(1)> -> !cir.ptr<!rec_A, target_address_space(1)>

// LLVM-LABEL: define {{.*}}@_Z11base_memberPU3AS11D
// LLVM: %[[A:.*]] = getelementptr inbounds nuw %struct.A, ptr addrspace(1) %{{.*}}, i32 0, i32 0
// LLVM: load i32, ptr addrspace(1) %[[A]]

// OGCG-LABEL: define {{.*}}@_Z11base_memberPU3AS11D
// OGCG: %[[A:.*]] = getelementptr inbounds nuw %struct.A, ptr addrspace(1) %{{.*}}, i32 0, i32 0
// OGCG: load i32, ptr addrspace(1) %[[A]]

int nonzero_base_member(AS1 D *d) { return d->b; }

// CIR-LABEL: cir.func {{.*}}@_Z19nonzero_base_memberPU3AS11D
// CIR: cir.base_class_addr nonnull %{{.*}} [4] : !cir.ptr<!rec_D, target_address_space(1)> -> !cir.ptr<!rec_B, target_address_space(1)>

// LLVM-LABEL: define {{.*}}@_Z19nonzero_base_memberPU3AS11D
// LLVM: %[[BASE:.*]] = getelementptr i8, ptr addrspace(1) %{{.*}}, i32 4
// LLVM: %[[B:.*]] = getelementptr inbounds nuw %struct.B, ptr addrspace(1) %[[BASE]], i32 0, i32 0
// LLVM: load i32, ptr addrspace(1) %[[B]]

// OGCG-LABEL: define {{.*}}@_Z19nonzero_base_memberPU3AS11D
// OGCG: %[[BASE:.*]] = getelementptr inbounds i8, ptr addrspace(1) %{{.*}}, i64 4
// OGCG: %[[B:.*]] = getelementptr inbounds nuw %struct.B, ptr addrspace(1) %[[BASE]], i32 0, i32 0
// OGCG: load i32, ptr addrspace(1) %[[B]]

AS1 B *to_base(AS1 D *d) { return d; }

// CIR-LABEL: cir.func {{.*}}@_Z7to_basePU3AS11D
// CIR: cir.base_class_addr %{{.*}} [4] : !cir.ptr<!rec_D, target_address_space(1)> -> !cir.ptr<!rec_B, target_address_space(1)>

// LLVM-LABEL: define {{.*}}@_Z7to_basePU3AS11D
// LLVM: %[[ISNULL:.*]] = icmp eq ptr addrspace(1) %[[D:.*]], null
// LLVM: %[[BASE:.*]] = getelementptr i8, ptr addrspace(1) %[[D]], i32 4
// LLVM: select i1 %[[ISNULL]], ptr addrspace(1) %[[D]], ptr addrspace(1) %[[BASE]]

// OGCG-LABEL: define {{.*}}@_Z7to_basePU3AS11D
// OGCG: %[[BASE:.*]] = getelementptr inbounds i8, ptr addrspace(1) %{{.*}}, i64 4
// OGCG: phi ptr addrspace(1) [ %[[BASE]], %{{.*}} ], [ null, %{{.*}} ]

AS1 D *to_derived(AS1 B *b) { return static_cast<AS1 D *>(b); }

// CIR-LABEL: cir.func {{.*}}@_Z10to_derivedPU3AS11B
// CIR: cir.derived_class_addr %{{.*}} [4] : !cir.ptr<!rec_B, target_address_space(1)> -> !cir.ptr<!rec_D, target_address_space(1)>

// LLVM-LABEL: define {{.*}}@_Z10to_derivedPU3AS11B
// LLVM: %[[ISNULL:.*]] = icmp eq ptr addrspace(1) %[[B:.*]], null
// LLVM: %[[DERIVED:.*]] = getelementptr inbounds i8, ptr addrspace(1) %[[B]], i32 -4
// LLVM: select i1 %[[ISNULL]], ptr addrspace(1) %[[B]], ptr addrspace(1) %[[DERIVED]]

// OGCG-LABEL: define {{.*}}@_Z10to_derivedPU3AS11B
// OGCG: %[[DERIVED:.*]] = getelementptr inbounds i8, ptr addrspace(1) %{{.*}}, i64 -4
// OGCG: phi ptr addrspace(1) [ %[[DERIVED]], %{{.*}} ], [ null, %{{.*}} ]

#ifdef VBASE
struct V { int v; };
struct W : virtual V { int w; };

AS1 V *to_virtual_base(AS1 W *w) { return w; }

// CIR-VBASE-LABEL: cir.func {{.*}}@_Z15to_virtual_basePU3AS11W
// CIR-VBASE: cir.ternary
// CIR-VBASE:   cir.const #cir.ptr<null> : !cir.ptr<!rec_V, target_address_space(1)>
// CIR-VBASE: }, false {
// CIR-VBASE:   %[[BYTES:.*]] = cir.cast bitcast %{{.*}} : !cir.ptr<!rec_W, target_address_space(1)> -> !cir.ptr<!u8i, target_address_space(1)>
// CIR-VBASE:   cir.ptr_stride %[[BYTES]], %{{.*}} : (!cir.ptr<!u8i, target_address_space(1)>, !s64i) -> !cir.ptr<!u8i, target_address_space(1)>
// CIR-VBASE: }) : (!cir.bool) -> !cir.ptr<!rec_V, target_address_space(1)>
#endif
