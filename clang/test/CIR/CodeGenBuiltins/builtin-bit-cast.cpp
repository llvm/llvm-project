// RUN: %clang_cc1 -std=c++20 -fclangir -triple aarch64 -emit-cir %s  -o - | FileCheck --check-prefix=CIR %s
// RUN: %clang_cc1 -std=c++20 -fclangir -triple aarch64 -emit-llvm %s -o - | FileCheck --check-prefixes=LLVM,LLVM-VIA-CIR %s
// RUN: %clang_cc1 -std=c++20           -triple aarch64 -emit-llvm %s -o - | FileCheck --check-prefixes=LLVM,LLVM-DIRECT %s

//=============================================================================
// NOTES
//
// Major differences between code lowered via ClangIR and directly to LLVM
// (e.g. different return types) are captured by using LLVM-VIA-CIR and LLVM-DIRECT labels.
//
// Minor differences (e.g. presence of `noundef` attached to argumens, `align`
// attribute attached to pointers), look for catch-alls like {{.*}}.
//
//=============================================================================

float test_scalar(int &oper) {
  return __builtin_bit_cast(float, oper);
}

// CIR-LABEL: cir.func {{.*}} @_Z11test_scalarRi
//       CIR:   %[[#SRC_PTR:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
//  CIR-NEXT:   %[[#DST_PTR:]] = cir.cast bitcast %[[#SRC_PTR]] : !cir.ptr<!s32i> -> !cir.ptr<!cir.float>
//  CIR-NEXT:   %{{.+}} = cir.load{{.*}} %[[#DST_PTR]] : !cir.ptr<!cir.float>, !cir.float

// LLVM-LABEL: define dso_local{{.*}} float @_Z11test_scalarRi
//       LLVM:   %[[#PTR:]] = load ptr, ptr %{{.+}}, align 8
//  LLVM-NEXT:   %{{.+}} = load float, ptr %[[#PTR]], align 4

struct two_ints {
  int x;
  int y;
};

unsigned long test_aggregate_to_scalar(two_ints &ti) {
  return __builtin_bit_cast(unsigned long, ti);
}

// CIR-LABEL: cir.func {{.*}} @_Z24test_aggregate_to_scalarR8two_ints
//       CIR:   %[[#SRC_PTR:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!cir.ptr<!rec_two_ints>>, !cir.ptr<!rec_two_ints>
//  CIR-NEXT:   %[[#DST_PTR:]] = cir.cast bitcast %[[#SRC_PTR]] : !cir.ptr<!rec_two_ints> -> !cir.ptr<!u64i>
//  CIR-NEXT:   %{{.+}} = cir.load{{.*}} %[[#DST_PTR]] : !cir.ptr<!u64i>, !u64i

// LLVM-LABEL: define dso_local{{.*}} i64 @_Z24test_aggregate_to_scalarR8two_ints
//       LLVM:   %[[#PTR:]] = load ptr, ptr %{{.+}}, align 8
//  LLVM-NEXT:   %{{.+}} = load i64, ptr %[[#PTR]], align 4

struct two_floats {
  float x;
  float y;
};

two_floats test_aggregate_record(two_ints& ti) {
   return __builtin_bit_cast(two_floats, ti);
}

// CIR-LABEL: cir.func {{.*}} @_Z21test_aggregate_recordR8two_ints
//       CIR:   %[[#SRC_PTR:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!cir.ptr<!rec_two_ints>>, !cir.ptr<!rec_two_ints>
//  CIR-NEXT:   %[[#SRC_VOID_PTR:]] = cir.cast bitcast %[[#SRC_PTR]] : !cir.ptr<!rec_two_ints> -> !cir.ptr<!void>
//  CIR-NEXT:   %[[#DST_VOID_PTR:]] = cir.cast bitcast %{{.+}} : !cir.ptr<!rec_two_floats> -> !cir.ptr<!void>
//  CIR-NEXT:   %[[#SIZE:]] = cir.const #cir.int<8> : !u64i
//  CIR-NEXT:   cir.libc.memcpy %[[#SIZE]] bytes from %[[#SRC_VOID_PTR]] to %[[#DST_VOID_PTR]] : !u64i, !cir.ptr<!void> -> !cir.ptr<!void>

// LLVM-LABEL: define dso_local{{.*}} %struct.two_floats @_Z21test_aggregate_recordR8two_ints
//       LLVM:   %[[DST_SLOT:.*]] = alloca %struct.two_floats{{.*}}, align 4
//       LLVM:   %[[SRC_PTR:.*]] = load ptr, ptr {{.*}}, align 8
//  LLVM-NEXT:   call void @llvm.memcpy.p0.p0.i64(ptr{{.*}} %[[DST_SLOT]], ptr{{.*}} %[[SRC_PTR]], i64 8, i1 false)
//  LLVM-NEXT:   %{{.+}} = load %struct.two_floats, ptr %[[DST_SLOT]], align 4

two_floats test_aggregate_array(int (&ary)[2]) {
  return __builtin_bit_cast(two_floats, ary);
}

// CIR-LABEL: cir.func {{.*}} @_Z20test_aggregate_arrayRA2_i
//       CIR:   %[[#SRC_PTR:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!cir.ptr<!cir.array<!s32i x 2>>>, !cir.ptr<!cir.array<!s32i x 2>>
//  CIR-NEXT:   %[[#SRC_VOID_PTR:]] = cir.cast bitcast %[[#SRC_PTR]] : !cir.ptr<!cir.array<!s32i x 2>> -> !cir.ptr<!void>
//  CIR-NEXT:   %[[#DST_VOID_PTR:]] = cir.cast bitcast %{{.+}} : !cir.ptr<!rec_two_floats> -> !cir.ptr<!void>
//  CIR-NEXT:   %[[#SIZE:]] = cir.const #cir.int<8> : !u64i
//  CIR-NEXT:   cir.libc.memcpy %[[#SIZE]] bytes from %[[#SRC_VOID_PTR]] to %[[#DST_VOID_PTR]] : !u64i, !cir.ptr<!void> -> !cir.ptr<!void>

// LLVM-LABEL: define dso_local{{.*}} %struct.two_floats @_Z20test_aggregate_arrayRA2_i
//       LLVM:   %[[DST_SLOT:.*]] = alloca %struct.two_floats{{.*}}, align 4
//       LLVM:   %[[SRC_PTR:.*]] = load ptr, ptr {{.*}}, align 8
//  LLVM-NEXT:   call void @llvm.memcpy.p0.p0.i64(ptr{{.*}} %[[DST_SLOT]], ptr{{.*}} %[[SRC_PTR]], i64 8, i1 false)
//  LLVM-NEXT:   %{{.+}} = load %struct.two_floats, ptr %[[DST_SLOT]], align 4

two_ints test_scalar_to_aggregate(unsigned long ul) {
  return __builtin_bit_cast(two_ints, ul);
}

// CIR-LABEL: cir.func {{.*}} @_Z24test_scalar_to_aggregatem
//       CIR:   %[[#SRC_VOID_PTR:]] = cir.cast bitcast %{{.+}} : !cir.ptr<!u64i> -> !cir.ptr<!void>
//  CIR-NEXT:   %[[#DST_VOID_PTR:]] = cir.cast bitcast %{{.+}} : !cir.ptr<!rec_two_ints> -> !cir.ptr<!void>
//  CIR-NEXT:   %[[#SIZE:]] = cir.const #cir.int<8> : !u64i
//  CIR-NEXT:   cir.libc.memcpy %[[#SIZE]] bytes from %[[#SRC_VOID_PTR]] to %[[#DST_VOID_PTR]] : !u64i, !cir.ptr<!void> -> !cir.ptr<!void>

// LLVM-DIRECT-LABEL: define dso_local i64 @_Z24test_scalar_to_aggregatem
// LLVM-VIA-CIR-LABEL: define dso_local %struct.two_ints @_Z24test_scalar_to_aggregatem
//       LLVM:   %[[DST_SLOT:.*]] = alloca %struct.two_ints{{.*}}, align 4
//       LLVM:   call void @llvm.memcpy.p0.p0.i64(ptr{{.*}} %[[DST_SLOT]], ptr{{.*}} %{{.+}}, i64 8, i1 false)
//  LLVM-DIRECT-NEXT:   %{{.+}} = load i64, ptr %[[DST_SLOT]], align 4
//  LLVM-VIA-CIR-NEXT:   %{{.+}} = load %struct.two_ints, ptr %[[DST_SLOT]], align 4

unsigned long test_array(int (&ary)[2]) {
  return __builtin_bit_cast(unsigned long, ary);
}

// CIR-LABEL: cir.func {{.*}} @_Z10test_arrayRA2_i
//      CIR:   %[[#SRC_PTR:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!cir.ptr<!cir.array<!s32i x 2>>>, !cir.ptr<!cir.array<!s32i x 2>>
// CIR-NEXT:   %[[#DST_PTR:]] = cir.cast bitcast %[[#SRC_PTR]] : !cir.ptr<!cir.array<!s32i x 2>> -> !cir.ptr<!u64i>
// CIR-NEXT:   %{{.+}} = cir.load{{.*}} %[[#DST_PTR]] : !cir.ptr<!u64i>, !u64i

// LLVM-LABEL: define dso_local{{.*}} i64 @_Z10test_arrayRA2_i
//       LLVM:   %[[SRC_PTR:.*]] = load ptr, ptr %{{.+}}, align 8
//  LLVM-NEXT:   %{{.+}} = load i64, ptr %[[SRC_PTR]], align 4

two_ints test_rvalue_aggregate() {
  return __builtin_bit_cast(two_ints, 42ul);
}

// CIR-LABEL: cir.func {{.*}} @_Z21test_rvalue_aggregatev()
//       CIR:   cir.scope {
//  CIR-NEXT:     %[[#TMP_SLOT:]] = cir.alloca !u64i, !cir.ptr<!u64i>
//  CIR-NEXT:     %[[#A:]] = cir.const #cir.int<42> : !u64i
//  CIR-NEXT:     cir.store{{.*}} %[[#A]], %[[#TMP_SLOT]] : !u64i, !cir.ptr<!u64i>
//  CIR-NEXT:     %[[#SRC_VOID_PTR:]] = cir.cast bitcast %[[#TMP_SLOT]] : !cir.ptr<!u64i> -> !cir.ptr<!void>
//  CIR-NEXT:     %[[#DST_VOID_PTR:]] = cir.cast bitcast %0 : !cir.ptr<!rec_two_ints> -> !cir.ptr<!void>
//  CIR-NEXT:     %[[#SIZE:]] = cir.const #cir.int<8> : !u64i
//  CIR-NEXT:     cir.libc.memcpy %[[#SIZE]] bytes from %[[#SRC_VOID_PTR]] to %[[#DST_VOID_PTR]] : !u64i, !cir.ptr<!void> -> !cir.ptr<!void>
//  CIR-NEXT:   }

/// FIXME: The function signature below should be identical for both lowering
/// paths, but CIR is still missing calling convention lowering. Update this
/// once calling convention is unstreamed.

// LLVM-DIRECT-LABEL: define dso_local{{.*}} i64 @_Z21test_rvalue_aggregatev
// LLVM-VIA-CIR-LABEL: define dso_local{{.*}} %struct.two_ints @_Z21test_rvalue_aggregatev
//  LLVM:   %[[SRC_SLOT:.*]] = alloca i64{{.*}}, align 8
//  LLVM:   store i64 42, ptr %[[SRC_SLOT]], align 8
//  LLVM-NEXT:   call void @llvm.memcpy.p0.p0.i64(ptr{{.*}} %{{.+}}, ptr{{.*}} %[[SRC_SLOT]], i64 8, i1 false)
