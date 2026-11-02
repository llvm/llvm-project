// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

// A member is stored inside its record, so the pointer to it is in the address
// space of the record.

#define AS1 __attribute__((address_space(1)))

struct S { int a; int b : 3; };
union U { int i; float f; };
struct B { B(); int i; char c; };
struct O { [[no_unique_address]] B b; char d; };

int field(AS1 S *s) { return s->a; }

// CIR-LABEL: cir.func {{.*}}@_Z5fieldPU3AS11S
// CIR: cir.get_member %{{.*}}[0] {name = "a"} : !cir.ptr<!rec_S, target_address_space(1)> -> !cir.ptr<!s32i, target_address_space(1)>

// LLVM-LABEL: define {{.*}}@_Z5fieldPU3AS11S
// LLVM: %[[A:.*]] = getelementptr inbounds nuw %struct.S, ptr addrspace(1) %{{.*}}, i32 0, i32 0
// LLVM: load i32, ptr addrspace(1) %[[A]]

// OGCG-LABEL: define {{.*}}@_Z5fieldPU3AS11S
// OGCG: %[[A:.*]] = getelementptr inbounds nuw %struct.S, ptr addrspace(1) %{{.*}}, i32 0, i32 0
// OGCG: load i32, ptr addrspace(1) %[[A]]

void bitfield(AS1 S *s) { s->b = 2; }

// CIR-LABEL: cir.func {{.*}}@_Z8bitfieldPU3AS11S
// CIR: cir.get_member %{{.*}}[1] {name = "b"} : !cir.ptr<!rec_S, target_address_space(1)> -> !cir.ptr<!u8i, target_address_space(1)>

// LLVM-LABEL: define {{.*}}@_Z8bitfieldPU3AS11S
// LLVM: %[[B:.*]] = getelementptr inbounds nuw %struct.S, ptr addrspace(1) %{{.*}}, i32 0, i32 1
// LLVM: store i8 %{{.*}}, ptr addrspace(1) %[[B]]

// OGCG-LABEL: define {{.*}}@_Z8bitfieldPU3AS11S
// OGCG: %[[B:.*]] = getelementptr inbounds nuw %struct.S, ptr addrspace(1) %{{.*}}, i32 0, i32 1
// OGCG: store i8 %{{.*}}, ptr addrspace(1) %[[B]]

float union_member(AS1 U *u) { return u->f; }

// CIR-LABEL: cir.func {{.*}}@_Z12union_memberPU3AS11U
// CIR: cir.get_member %{{.*}}[1] {name = "f"} : !cir.ptr<!rec_U, target_address_space(1)> -> !cir.ptr<!cir.float, target_address_space(1)>

// LLVM-LABEL: define {{.*}}@_Z12union_memberPU3AS11U
// LLVM: %[[U:.*]] = load ptr addrspace(1), ptr
// LLVM: load float, ptr addrspace(1) %[[U]]

// OGCG-LABEL: define {{.*}}@_Z12union_memberPU3AS11U
// OGCG: %[[U:.*]] = load ptr addrspace(1), ptr
// OGCG: load float, ptr addrspace(1) %[[U]]

int overlapping(AS1 O *o) { return o->b.i; }

// CIR-LABEL: cir.func {{.*}}@_Z11overlappingPU3AS11O
// CIR: %[[BASE:.*]] = cir.get_member %{{.*}}[0] {name = "b"} : !cir.ptr<!rec_O, target_address_space(1)> -> !cir.ptr<!rec_B2Ebase, target_address_space(1)>
// CIR: cir.cast bitcast %[[BASE]] : !cir.ptr<!rec_B2Ebase, target_address_space(1)> -> !cir.ptr<!rec_B, target_address_space(1)>

// LLVM-LABEL: define {{.*}}@_Z11overlappingPU3AS11O
// LLVM: %[[OB:.*]] = getelementptr inbounds nuw %struct.O, ptr addrspace(1) %{{.*}}, i32 0, i32 0
// LLVM: %[[I:.*]] = getelementptr inbounds nuw %struct.B, ptr addrspace(1) %[[OB]], i32 0, i32 0
// LLVM: load i32, ptr addrspace(1) %[[I]]

// OGCG-LABEL: define {{.*}}@_Z11overlappingPU3AS11O
// OGCG: %[[OB:.*]] = getelementptr inbounds nuw %struct.O, ptr addrspace(1) %{{.*}}, i32 0, i32 0
// OGCG: %[[I:.*]] = getelementptr inbounds nuw %struct.B, ptr addrspace(1) %[[OB]], i32 0, i32 0
// OGCG: load i32, ptr addrspace(1) %[[I]]
