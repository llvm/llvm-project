// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir --check-prefix=CIR %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll --check-prefix=LLVM %s

float test_scalar(int &oper) {
  return __builtin_bit_cast(float, oper);
}

// CIR-LABEL: cir.func @_Z11test_scalarRi
//       CIR:   %[[#SRC_PTR:]] = cir.load %{{.+}} : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
//  CIR-NEXT:   %[[#DST_PTR:]] = cir.cast(bitcast, %[[#SRC_PTR]] : !cir.ptr<!s32i>), !cir.ptr<!cir.float>
//  CIR-NEXT:   %{{.+}} = cir.load %[[#DST_PTR]] : !cir.ptr<!cir.float>, !cir.float
//       CIR: }

// LLVM-LABEL: define dso_local float @_Z11test_scalarRi
//       LLVM:   %[[#PTR:]] = load ptr, ptr %{{.+}}, align 8
//  LLVM-NEXT:   %{{.+}} = load float, ptr %[[#PTR]], align 4
//       LLVM: }

struct two_ints {
  int x;
  int y;
};

unsigned long test_aggregate_to_scalar(two_ints &ti) {
  return __builtin_bit_cast(unsigned long, ti);
}

// CIR-LABEL: cir.func @_Z24test_aggregate_to_scalarR8two_ints
//       CIR:   %[[#SRC_PTR:]] = cir.load %{{.+}} : !cir.ptr<!cir.ptr<!ty_two_ints>>, !cir.ptr<!ty_two_ints>
//  CIR-NEXT:   %[[#DST_PTR:]] = cir.cast(bitcast, %[[#SRC_PTR]] : !cir.ptr<!ty_two_ints>), !cir.ptr<!u64i>
//  CIR-NEXT:   %{{.+}} = cir.load %[[#DST_PTR]] : !cir.ptr<!u64i>, !u64i
//       CIR: }

// LLVM-LABEL: define dso_local i64 @_Z24test_aggregate_to_scalarR8two_ints
//       LLVM:   %[[#PTR:]] = load ptr, ptr %{{.+}}, align 8
//  LLVM-NEXT:   %{{.+}} = load i64, ptr %[[#PTR]], align 8
//       LLVM: }

struct two_floats {
  float x;
  float y;
};

two_floats test_aggregate_record(two_ints& ti) {
   return __builtin_bit_cast(two_floats, ti);
}

// CIR-LABEL: cir.func @_Z21test_aggregate_recordR8two_ints
//       CIR:   %[[#SRC_PTR:]] = cir.load %{{.+}} : !cir.ptr<!cir.ptr<!ty_two_ints>>, !cir.ptr<!ty_two_ints>
//  CIR-NEXT:   %[[#SRC_VOID_PTR:]] = cir.cast(bitcast, %[[#SRC_PTR]] : !cir.ptr<!ty_two_ints>), !cir.ptr<!void>
//  CIR-NEXT:   %[[#DST_VOID_PTR:]] = cir.cast(bitcast, %{{.+}} : !cir.ptr<!ty_two_floats>), !cir.ptr<!void>
//  CIR-NEXT:   %[[#SIZE:]] = cir.const #cir.int<8> : !u64i
//  CIR-NEXT:   cir.libc.memcpy %[[#SIZE]] bytes from %[[#SRC_VOID_PTR]] to %[[#DST_VOID_PTR]] : !u64i, !cir.ptr<!void> -> !cir.ptr<!void>
//       CIR: }

// LLVM-LABEL: define dso_local %struct.two_floats @_Z21test_aggregate_recordR8two_ints
//       LLVM:   %[[#DST_SLOT:]] = alloca %struct.two_floats, i64 1, align 4
//       LLVM:   %[[#SRC_PTR:]] = load ptr, ptr %2, align 8
//  LLVM-NEXT:   call void @llvm.memcpy.p0.p0.i64(ptr %[[#DST_SLOT]], ptr %[[#SRC_PTR]], i64 8, i1 false)
//  LLVM-NEXT:   %{{.+}} = load %struct.two_floats, ptr %[[#DST_SLOT]], align 4
//       LLVM: }

two_floats test_aggregate_array(int (&ary)[2]) {
  return __builtin_bit_cast(two_floats, ary);
}

// CIR-LABEL: cir.func @_Z20test_aggregate_arrayRA2_i
//       CIR:   %[[#SRC_PTR:]] = cir.load %{{.+}} : !cir.ptr<!cir.ptr<!cir.array<!s32i x 2>>>, !cir.ptr<!cir.array<!s32i x 2>>
//  CIR-NEXT:   %[[#SRC_VOID_PTR:]] = cir.cast(bitcast, %[[#SRC_PTR]] : !cir.ptr<!cir.array<!s32i x 2>>), !cir.ptr<!void>
//  CIR-NEXT:   %[[#DST_VOID_PTR:]] = cir.cast(bitcast, %{{.+}} : !cir.ptr<!ty_two_floats>), !cir.ptr<!void>
//  CIR-NEXT:   %[[#SIZE:]] = cir.const #cir.int<8> : !u64i
//  CIR-NEXT:   cir.libc.memcpy %[[#SIZE]] bytes from %[[#SRC_VOID_PTR]] to %[[#DST_VOID_PTR]] : !u64i, !cir.ptr<!void> -> !cir.ptr<!void>
//       CIR: }

// LLVM-LABEL: define dso_local %struct.two_floats @_Z20test_aggregate_arrayRA2_i
//       LLVM:   %[[#DST_SLOT:]] = alloca %struct.two_floats, i64 1, align 4
//       LLVM:   %[[#SRC_PTR:]] = load ptr, ptr %2, align 8
//  LLVM-NEXT:   call void @llvm.memcpy.p0.p0.i64(ptr %[[#DST_SLOT]], ptr %[[#SRC_PTR]], i64 8, i1 false)
//  LLVM-NEXT:   %{{.+}} = load %struct.two_floats, ptr %[[#DST_SLOT]], align 4
//       LLVM: }

two_ints test_scalar_to_aggregate(unsigned long ul) {
  return __builtin_bit_cast(two_ints, ul);
}

// CIR-LABEL: cir.func @_Z24test_scalar_to_aggregatem
//       CIR:   %[[#SRC_VOID_PTR:]] = cir.cast(bitcast, %{{.+}} : !cir.ptr<!u64i>), !cir.ptr<!void>
//  CIR-NEXT:   %[[#DST_VOID_PTR:]] = cir.cast(bitcast, %{{.+}} : !cir.ptr<!ty_two_ints>), !cir.ptr<!void>
//  CIR-NEXT:   %[[#SIZE:]] = cir.const #cir.int<8> : !u64i
//  CIR-NEXT:   cir.libc.memcpy %[[#SIZE]] bytes from %[[#SRC_VOID_PTR]] to %[[#DST_VOID_PTR]] : !u64i, !cir.ptr<!void> -> !cir.ptr<!void>
//       CIR: }

// LLVM-LABEL: define dso_local %struct.two_ints @_Z24test_scalar_to_aggregatem
//       LLVM:   %[[#DST_SLOT:]] = alloca %struct.two_ints, i64 1, align 4
//       LLVM:   call void @llvm.memcpy.p0.p0.i64(ptr %[[#DST_SLOT]], ptr %{{.+}}, i64 8, i1 false)
//  LLVM-NEXT:   %{{.+}} = load %struct.two_ints, ptr %[[#DST_SLOT]], align 4
//       LLVM: }

unsigned long test_array(int (&ary)[2]) {
  return __builtin_bit_cast(unsigned long, ary);
}

// CIR-LABEL: cir.func @_Z10test_arrayRA2_i
//      CIR:   %[[#SRC_PTR:]] = cir.load %{{.+}} : !cir.ptr<!cir.ptr<!cir.array<!s32i x 2>>>, !cir.ptr<!cir.array<!s32i x 2>>
// CIR-NEXT:   %[[#DST_PTR:]] = cir.cast(bitcast, %[[#SRC_PTR]] : !cir.ptr<!cir.array<!s32i x 2>>), !cir.ptr<!u64i>
// CIR-NEXT:   %{{.+}} = cir.load %[[#DST_PTR]] : !cir.ptr<!u64i>, !u64i
//      CIR: }

// LLVM-LABEL: define dso_local i64 @_Z10test_arrayRA2_i
//       LLVM:   %[[#SRC_PTR:]] = load ptr, ptr %{{.+}}, align 8
//  LLVM-NEXT:   %{{.+}} = load i64, ptr %[[#SRC_PTR]], align 8
//       LLVM: }

two_ints test_rvalue_aggregate() {
  return __builtin_bit_cast(two_ints, 42ul);
}

// CIR-LABEL: cir.func @_Z21test_rvalue_aggregatev()
//       CIR:   cir.scope {
//  CIR-NEXT:     %[[#TMP_SLOT:]] = cir.alloca !u64i, !cir.ptr<!u64i>
//  CIR-NEXT:     %[[#A:]] = cir.const #cir.int<42> : !u64i
//  CIR-NEXT:     cir.store %[[#A]], %[[#TMP_SLOT]] : !u64i, !cir.ptr<!u64i>
//  CIR-NEXT:     %[[#SRC_VOID_PTR:]] = cir.cast(bitcast, %[[#TMP_SLOT]] : !cir.ptr<!u64i>), !cir.ptr<!void>
//  CIR-NEXT:     %[[#DST_VOID_PTR:]] = cir.cast(bitcast, %0 : !cir.ptr<!ty_two_ints>), !cir.ptr<!void>
//  CIR-NEXT:     %[[#SIZE:]] = cir.const #cir.int<8> : !u64i
//  CIR-NEXT:     cir.libc.memcpy %[[#SIZE]] bytes from %[[#SRC_VOID_PTR]] to %[[#DST_VOID_PTR]] : !u64i, !cir.ptr<!void> -> !cir.ptr<!void>
//  CIR-NEXT:   }
//       CIR: }

// LLVM-LABEL: define dso_local %struct.two_ints @_Z21test_rvalue_aggregatev
//       LLVM:   %[[#SRC_SLOT:]] = alloca i64, i64 1, align 8
//  LLVM-NEXT:   store i64 42, ptr %[[#SRC_SLOT]], align 8
//  LLVM-NEXT:   call void @llvm.memcpy.p0.p0.i64(ptr %{{.+}}, ptr %[[#SRC_SLOT]], i64 8, i1 false)
//       LLVM: }
