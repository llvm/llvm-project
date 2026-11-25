// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// A struct wider than 64 bits is passed as the fields of its LLVM type. A
// bit-field access unit is passed as its storage type, and a zero-width
// bit-field, which has no storage, is dropped.

typedef struct { int a, b; unsigned x : 16, y : 11; char c; } bf_t;
typedef struct { int a, b; unsigned x : 3; unsigned : 0; char c; } zw_t;
typedef struct __attribute__((packed)) { char c; int a[2]; unsigned x : 12; } pk_t;

// CIR-DAG: ![[FLAT:rec_anon_struct[0-9]*]] = !cir.struct<{data !s32i, data !s32i, data !u32i, data !s8i}>
// CIR-DAG: ![[ZW_FLAT:rec_anon_struct[0-9]*]] = !cir.struct<{data !s32i, data !s32i, data !u8i, data !cir.array<!u8i x 3>, data !s8i}>
// CIR-DAG: ![[PK_FLAT:rec_anon_struct[0-9]*]] = !cir.struct<packed {data !s8i, data !cir.array<!s32i x 2>, data !u16i}>
// CIR-DAG: !rec_bf_t = !cir.struct<"bf_t" {data !s32i, data !s32i, bitfield !cir.bitfield<!u32i, [#cir.bitfield_decl<!u32i, 16>, #cir.bitfield_decl<!u32i, 11>]>, data !s8i}>
// CIR-DAG: !rec_zw_t = !cir.struct<"zw_t" {data !s32i, data !s32i, bitfield !cir.bitfield<!u8i, [#cir.bitfield_decl<!u32i, 3>]>, pad !cir.array<!u8i x 3>, empty !cir.bitfield<[#cir.bitfield_decl<!u32i, 0, unnamed>]>, data !s8i}>
// CIR-DAG: !rec_pk_t = !cir.struct<"pk_t" packed {data !s8i, data !cir.array<!s32i x 2>, bitfield !cir.bitfield<!u16i, [#cir.bitfield_decl<!u32i, 12>]>}>

void arg_bf(bf_t s) {}

// CIR:      cir.func{{.*}} @arg_bf(%[[A:[^:]+]]: !s32i loc{{.*}}, %[[B:[^:]+]]: !s32i loc{{.*}}, %[[UNIT:[^:]+]]: !u32i loc{{.*}}, %[[C:[^:]+]]: !s8i loc{{.*}})
// CIR-NEXT:   %[[COERCE:.*]] = cir.alloca "coerce" align(4) : !cir.ptr<![[FLAT]], target_address_space(5)>
// CIR-NEXT:   %[[F0:.*]] = cir.get_member %[[COERCE]][0] {name = ""} : !cir.ptr<![[FLAT]], target_address_space(5)> -> !cir.ptr<!s32i, target_address_space(5)>
// CIR-NEXT:   cir.store %[[A]], %[[F0]] : !s32i, !cir.ptr<!s32i, target_address_space(5)>
// CIR-NEXT:   %[[F1:.*]] = cir.get_member %[[COERCE]][1] {name = ""} : !cir.ptr<![[FLAT]], target_address_space(5)> -> !cir.ptr<!s32i, target_address_space(5)>
// CIR-NEXT:   cir.store %[[B]], %[[F1]] : !s32i, !cir.ptr<!s32i, target_address_space(5)>
// CIR-NEXT:   %[[F2:.*]] = cir.get_member %[[COERCE]][2] {name = ""} : !cir.ptr<![[FLAT]], target_address_space(5)> -> !cir.ptr<!u32i, target_address_space(5)>
// CIR-NEXT:   cir.store %[[UNIT]], %[[F2]] : !u32i, !cir.ptr<!u32i, target_address_space(5)>
// CIR-NEXT:   %[[F3:.*]] = cir.get_member %[[COERCE]][3] {name = ""} : !cir.ptr<![[FLAT]], target_address_space(5)> -> !cir.ptr<!s8i, target_address_space(5)>
// CIR-NEXT:   cir.store %[[C]], %[[F3]] : !s8i, !cir.ptr<!s8i, target_address_space(5)>
// CIR-NEXT:   %[[AS_BF:.*]] = cir.cast bitcast %[[COERCE]] : !cir.ptr<![[FLAT]], target_address_space(5)> -> !cir.ptr<!rec_bf_t, target_address_space(5)>
// CIR-NEXT:   %[[S:.*]] = cir.alloca "s" align(4) init : !cir.ptr<!rec_bf_t, target_address_space(5)>
// CIR-NEXT:   %[[S_VIEW:.*]] = cir.cast address_space %[[S]] : !cir.ptr<!rec_bf_t, target_address_space(5)> -> !cir.ptr<!rec_bf_t>
// CIR-NEXT:   cir.copy %[[AS_BF]] align(4) to %[[S_VIEW]] align(4) : !cir.ptr<!rec_bf_t, target_address_space(5)>, !cir.ptr<!rec_bf_t>
// CIR-NEXT:   cir.return

// LLVM: define dso_local void @arg_bf(i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i8 %{{.*}})

// The caller passes the access unit read from the struct.
void call_bf(bf_t *p) { arg_bf(*p); }

// CIR:      cir.func{{.*}} @call_bf(%[[P:[^:]+]]: !cir.ptr<!rec_bf_t>
// CIR-NEXT:   %[[COERCE:.*]] = cir.alloca "coerce" align(4) : !cir.ptr<!rec_bf_t, target_address_space(5)>
// CIR-NEXT:   %[[P_ADDR:.*]] = cir.alloca "p" align(8) init : !cir.ptr<!cir.ptr<!rec_bf_t>, target_address_space(5)>
// CIR-NEXT:   %[[P_VIEW:.*]] = cir.cast address_space %[[P_ADDR]] : !cir.ptr<!cir.ptr<!rec_bf_t>, target_address_space(5)> -> !cir.ptr<!cir.ptr<!rec_bf_t>>
// CIR-NEXT:   cir.store %[[P]], %[[P_VIEW]] : !cir.ptr<!rec_bf_t>, !cir.ptr<!cir.ptr<!rec_bf_t>>
// CIR-NEXT:   %[[SRC:.*]] = cir.load deref align(8) %[[P_VIEW]] : !cir.ptr<!cir.ptr<!rec_bf_t>>, !cir.ptr<!rec_bf_t>
// CIR-NEXT:   cir.copy %[[SRC]] align(4) to %[[COERCE]] align(4) : !cir.ptr<!rec_bf_t>, !cir.ptr<!rec_bf_t, target_address_space(5)>
// CIR-NEXT:   %[[AS_FLAT:.*]] = cir.cast bitcast %[[COERCE]] : !cir.ptr<!rec_bf_t, target_address_space(5)> -> !cir.ptr<![[FLAT]], target_address_space(5)>
// CIR-NEXT:   %[[F0:.*]] = cir.get_member %[[AS_FLAT]][0] {name = ""} : !cir.ptr<![[FLAT]], target_address_space(5)> -> !cir.ptr<!s32i, target_address_space(5)>
// CIR-NEXT:   %[[A:.*]] = cir.load %[[F0]] : !cir.ptr<!s32i, target_address_space(5)>, !s32i
// CIR-NEXT:   %[[F1:.*]] = cir.get_member %[[AS_FLAT]][1] {name = ""} : !cir.ptr<![[FLAT]], target_address_space(5)> -> !cir.ptr<!s32i, target_address_space(5)>
// CIR-NEXT:   %[[B:.*]] = cir.load %[[F1]] : !cir.ptr<!s32i, target_address_space(5)>, !s32i
// CIR-NEXT:   %[[F2:.*]] = cir.get_member %[[AS_FLAT]][2] {name = ""} : !cir.ptr<![[FLAT]], target_address_space(5)> -> !cir.ptr<!u32i, target_address_space(5)>
// CIR-NEXT:   %[[UNIT:.*]] = cir.load %[[F2]] : !cir.ptr<!u32i, target_address_space(5)>, !u32i
// CIR-NEXT:   %[[F3:.*]] = cir.get_member %[[AS_FLAT]][3] {name = ""} : !cir.ptr<![[FLAT]], target_address_space(5)> -> !cir.ptr<!s8i, target_address_space(5)>
// CIR-NEXT:   %[[C:.*]] = cir.load %[[F3]] : !cir.ptr<!s8i, target_address_space(5)>, !s8i
// CIR-NEXT:   cir.call @arg_bf(%[[A]], %[[B]], %[[UNIT]], %[[C]]) {convergent} : (!s32i, !s32i, !u32i, !s8i) -> ()
// CIR-NEXT:   cir.return

// LLVM: define dso_local void @call_bf(ptr noundef %{{.*}})
// LLVM:   call void @arg_bf(i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i8 %{{.*}})

// The zero-width bit-field is dropped, the padding ahead of it is passed.
void arg_zw(zw_t s) {}

// CIR:      cir.func{{.*}} @arg_zw(%{{[^:]+}}: !s32i loc{{.*}}, %{{[^:]+}}: !s32i loc{{.*}}, %{{[^:]+}}: !u8i loc{{.*}}, %{{[^:]+}}: !cir.array<!u8i x 3> loc{{.*}}, %{{[^:]+}}: !s8i loc{{.*}})
// CIR-NEXT:   %{{.*}} = cir.alloca "coerce" align(4) : !cir.ptr<![[ZW_FLAT]], target_address_space(5)>

// LLVM: define dso_local void @arg_zw(i32 %{{.*}}, i32 %{{.*}}, i8 %{{.*}}, [3 x i8] %{{.*}}, i8 %{{.*}})

// The flattened struct stays packed.
void arg_pk(pk_t s) {}

// CIR:      cir.func{{.*}} @arg_pk(%{{[^:]+}}: !s8i loc{{.*}}, %{{[^:]+}}: !cir.array<!s32i x 2> loc{{.*}}, %{{[^:]+}}: !u16i loc{{.*}})
// CIR-NEXT:   %{{.*}} = cir.alloca "coerce" align(1) : !cir.ptr<![[PK_FLAT]], target_address_space(5)>

// LLVM: define dso_local void @arg_pk(i8 %{{.*}}, [2 x i32] %{{.*}}, i16 %{{.*}})
