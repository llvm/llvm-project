// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// A record with a flexible array member is passed and returned in memory.
// A zero-length array is not a flexible array member and stays in registers.

typedef struct { int n; int data[]; } fam_t;
typedef struct { int a, b, c; int data[]; } fam_big_t;
typedef struct { int n; int data[0]; } zla_t;
typedef struct { int x; fam_t f; } fam_nested_t;

// The nested record gets the flag from its last field. zla_t does not.
// CIR: cir.record_layouts = {
// CIR-SAME: fam_big_t = #cir.record_layout<{{[^>]*}}, flexible_array_member = true>
// CIR-SAME: fam_nested_t = #cir.record_layout<{{[^>]*}}, flexible_array_member = true>
// CIR-SAME: fam_t = #cir.record_layout<{{[^>]*}}, flexible_array_member = true>
// CIR-SAME: zla_t = #cir.record_layout<arg_passing_kind = can_pass_in_regs, has_trivial_dtor = true, record_align = 4>}

void arg_fam(fam_t s) {}

// CIR:      cir.func {{.*}} @arg_fam(%[[S_ARG:[^:]+]]: !cir.ptr<!rec_fam_t> {llvm.align = 8 : i64, llvm.byval = !rec_fam_t, llvm.noundef}
// CIR-NEXT:   %[[S:.*]] = cir.alloca "s" align(4) init : !cir.ptr<!rec_fam_t>
// CIR-NEXT:   cir.copy %[[S_ARG]] align(8) to %[[S]] align(4) : !cir.ptr<!rec_fam_t>
// CIR-NEXT:   cir.return implicit

// LLVM: define dso_local void @arg_fam(ptr noundef byval(%struct.fam_t) align 8 %{{.*}})

// Without the flag, 12 bytes would be split into i64 and i32.
void arg_fam_big(fam_big_t s) {}

// CIR:      cir.func {{.*}} @arg_fam_big(%[[S_ARG:[^:]+]]: !cir.ptr<!rec_fam_big_t> {llvm.align = 8 : i64, llvm.byval = !rec_fam_big_t, llvm.noundef}
// CIR-NEXT:   %[[S:.*]] = cir.alloca "s" align(4) init : !cir.ptr<!rec_fam_big_t>
// CIR-NEXT:   cir.copy %[[S_ARG]] align(8) to %[[S]] align(4) : !cir.ptr<!rec_fam_big_t>
// CIR-NEXT:   cir.return implicit

// LLVM: define dso_local void @arg_fam_big(ptr noundef byval(%struct.fam_big_t) align 8 %{{.*}})

void arg_zla(zla_t s) {}

// CIR:      cir.func {{.*}} @arg_zla(%[[S_ARG:[^:]+]]: !s32i
// CIR-NEXT:   %[[COERCE:.*]] = cir.alloca "coerce" align(4) : !cir.ptr<!s32i>
// CIR-NEXT:   cir.store %[[S_ARG]], %[[COERCE]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT:   %[[COERCE_REC:.*]] = cir.cast bitcast %[[COERCE]] : !cir.ptr<!s32i> -> !cir.ptr<!rec_zla_t>
// CIR-NEXT:   %[[S:.*]] = cir.alloca "s" align(4) init : !cir.ptr<!rec_zla_t>
// CIR-NEXT:   cir.copy %[[COERCE_REC]] align(4) to %[[S]] align(4) : !cir.ptr<!rec_zla_t>
// CIR-NEXT:   cir.return implicit

// LLVM: define dso_local void @arg_zla(i32 %{{.*}})

void arg_fam_nested(fam_nested_t s) {}

// CIR:      cir.func {{.*}} @arg_fam_nested(%[[S_ARG:[^:]+]]: !cir.ptr<!rec_fam_nested_t> {llvm.align = 8 : i64, llvm.byval = !rec_fam_nested_t, llvm.noundef}
// CIR-NEXT:   %[[S:.*]] = cir.alloca "s" align(4) init : !cir.ptr<!rec_fam_nested_t>
// CIR-NEXT:   cir.copy %[[S_ARG]] align(8) to %[[S]] align(4) : !cir.ptr<!rec_fam_nested_t>
// CIR-NEXT:   cir.return implicit

// LLVM: define dso_local void @arg_fam_nested(ptr noundef byval(%struct.fam_nested_t) align 8 %{{.*}})

fam_t ret_fam(fam_t s) { return s; }

// CIR:      cir.func {{.*}} @ret_fam(%[[RET:[^:]+]]: !cir.ptr<!rec_fam_t> {llvm.align = 4 : i64, llvm.dead_on_unwind, llvm.noalias, llvm.sret = !rec_fam_t, llvm.writable} {{.*}}, %[[S_ARG:[^:]+]]: !cir.ptr<!rec_fam_t> {llvm.align = 8 : i64, llvm.byval = !rec_fam_t, llvm.noundef}
// CIR-NEXT:   %[[S:.*]] = cir.alloca "s" align(4) init : !cir.ptr<!rec_fam_t>
// CIR-NEXT:   cir.copy %[[S_ARG]] align(8) to %[[S]] align(4) : !cir.ptr<!rec_fam_t>
// CIR-NEXT:   cir.copy %[[S]] align(4) to %[[RET]] align(4) : !cir.ptr<!rec_fam_t>
// CIR-NEXT:   cir.return

// LLVM: define dso_local void @ret_fam(ptr dead_on_unwind noalias writable sret(%struct.fam_t) align 4 %{{.*}}, ptr noundef byval(%struct.fam_t) align 8 %{{.*}})

void call_fam(fam_t s) { arg_fam(s); }

// CIR:      cir.func {{.*}} @call_fam(%[[S_ARG:[^:]+]]: !cir.ptr<!rec_fam_t> {llvm.align = 8 : i64, llvm.byval = !rec_fam_t, llvm.noundef}
// CIR-NEXT:   %[[S:.*]] = cir.alloca "s" align(4) init : !cir.ptr<!rec_fam_t>
// CIR-NEXT:   cir.copy %[[S_ARG]] align(8) to %[[S]] align(4) : !cir.ptr<!rec_fam_t>
// CIR-NEXT:   %[[TMP:.*]] = cir.alloca "byval" align(8) : !cir.ptr<!rec_fam_t>
// CIR-NEXT:   cir.copy %[[S]] align(4) to %[[TMP]] align(8) : !cir.ptr<!rec_fam_t>
// CIR-NEXT:   cir.call @arg_fam(%[[TMP]]) : (!cir.ptr<!rec_fam_t> {llvm.align = 8 : i64, llvm.byval = !rec_fam_t, llvm.noundef}) -> ()
// CIR-NEXT:   cir.return implicit

// LLVM: define dso_local void @call_fam(ptr noundef byval(%struct.fam_t) align 8 %{{.*}})
// LLVM:   call void @arg_fam(ptr noundef byval(%struct.fam_t) align 8 %{{.*}})
