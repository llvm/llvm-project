// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir -O1
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll -O1
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll -O1 -relaxed-aliasing
// RUN: FileCheck --check-prefix=NO-TBAA --input-file=%t.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll -O0
// RUN: FileCheck --check-prefix=NO-TBAA --input-file=%t.ll %s

// NO-TBAA-NOT: !tbaa

// CIR: #tbaa[[CHAR:.*]] = #cir.tbaa_omnipotent_char
// CIR: #tbaa[[FLOAT:.*]] = #cir.tbaa_scalar<id = "float", type = !cir.float>
// CIR: #tbaa[[DOUBLE:.*]] = #cir.tbaa_scalar<id = "double", type = !cir.double>
// CIR: #tbaa[[LONG_DOUBLE:.*]] = #cir.tbaa_scalar<id = "long double", type = !cir.long_double<!cir.f80>>
// CIR: #tbaa[[INT:.*]] = #cir.tbaa_scalar<id = "int", type = !s32i>
// CIR: #tbaa[[LONG:.*]] = #cir.tbaa_scalar<id = "long", type = !s64i>
// CIR: #tbaa[[LONG_LONG:.*]] = #cir.tbaa_scalar<id = "long long", type = !s64i>

void test_int_and_float(int *a, float *b) {
  // CIR-LABEL: cir.func @test_int_and_float
  // CIR: cir.scope
  // CIR: %[[TMP1:.*]] = cir.load deref %{{.*}} : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
  // CIR: %[[TMP2:.*]] = cir.load %[[TMP1]] : !cir.ptr<!s32i>, !s32i tbaa(#tbaa[[INT]])
  // CIR: cir.if
  // CIR: %[[C2:.*]] = cir.const #cir.fp<2
  // CIR: %[[TMP3:.*]] = cir.load deref %[[ARG_b:.*]] : !cir.ptr<!cir.ptr<!cir.float>>, !cir.ptr<!cir.float>
  // CIR: cir.store %[[C2]], %[[TMP3]] : !cir.float, !cir.ptr<!cir.float> tbaa(#tbaa[[FLOAT]])
  // CIR: else
  // CIR: %[[C3:.*]] = cir.const #cir.fp<3
  // CIR: %[[TMP4:.*]] = cir.load deref %[[ARG_b]] : !cir.ptr<!cir.ptr<!cir.float>>, !cir.ptr<!cir.float>
  // CIR: cir.store %[[C3]], %[[TMP4]] : !cir.float, !cir.ptr<!cir.float> tbaa(#tbaa[[FLOAT]])

  // LLVM-LABEL: void @test_int_and_float
  // LLVM: %[[ARG_a:.*]] = load i32, ptr %{{.*}}, align 4, !tbaa ![[TBAA_INT:.*]]
  // LLVM: %[[COND:.*]] = icmp eq i32 %[[ARG_a]], 1
  // LLVM: %[[RET:.*]] = select i1 %[[COND]], float 2.000000e+00, float 3.000000e+00
  // LLVM: store float %[[RET]], ptr %{{.*}}, align 4, !tbaa ![[TBAA_FLOAT:.*]]
  // LLVM: ret void
  if (*a == 1) {
    *b = 2.0f;
  } else {
    *b = 3.0f;
  }
}

void test_long_and_double(long *a, double *b) {
  // CIR-LABEL: cir.func @test_long_and_double
  // CIR: cir.scope
  // CIR: %[[TMP1:.*]] = cir.load deref %{{.*}} : !cir.ptr<!cir.ptr<!s64i>>, !cir.ptr<!s64i>
  // CIR: %[[TMP2:.*]] = cir.load %[[TMP1]] : !cir.ptr<!s64i>, !s64i tbaa(#tbaa[[LONG]])
  // CIR: cir.if
  // CIR: %[[C2:.*]] = cir.const #cir.fp<2
  // CIR: %[[TMP3:.*]] = cir.load deref %[[ARG_b:.*]] : !cir.ptr<!cir.ptr<!cir.double>>, !cir.ptr<!cir.double>
  // CIR: cir.store %[[C2]], %[[TMP3]] : !cir.double, !cir.ptr<!cir.double> tbaa(#tbaa[[DOUBLE]])
  // CIR: else
  // CIR: %[[C3:.*]] = cir.const #cir.fp<3
  // CIR: %[[TMP4:.*]] = cir.load deref %[[ARG_b]] : !cir.ptr<!cir.ptr<!cir.double>>, !cir.ptr<!cir.double>
  // CIR: cir.store %[[C3]], %[[TMP4]] : !cir.double, !cir.ptr<!cir.double> tbaa(#tbaa[[DOUBLE]])

  // LLVM-LABEL: void @test_long_and_double
  // LLVM: %[[ARG_a:.*]] = load i64, ptr %{{.*}}, align 8, !tbaa ![[TBAA_LONG:.*]]
  // LLVM: %[[COND:.*]] = icmp eq i64 %[[ARG_a]], 1
  // LLVM: %[[RET:.*]] = select i1 %[[COND]], double 2.000000e+00, double 3.000000e+00
  // LLVM: store double %[[RET]], ptr %{{.*}}, align 8, !tbaa ![[TBAA_DOUBLE:.*]]
  // LLVM: ret void
  if (*a == 1L) {
    *b = 2.0;
  } else {
    *b = 3.0;
  }
}
void test_long_long_and_long_double(long long *a, long double *b) {
  // CIR-LABEL: cir.func @test_long_long_and_long_double
  // CIR: cir.scope
  // CIR: %[[TMP1:.*]] = cir.load deref %{{.*}} : !cir.ptr<!cir.ptr<!s64i>>, !cir.ptr<!s64i>
  // CIR: %[[TMP2:.*]] = cir.load %[[TMP1]] : !cir.ptr<!s64i>, !s64i tbaa(#tbaa[[LONG_LONG]])
  // CIR: cir.if
  // CIR: %[[C2:.*]] = cir.const #cir.fp<2
  // CIR: %[[TMP3:.*]] = cir.load deref %[[ARG_b:.*]] : !cir.ptr<!cir.ptr<!cir.long_double<!cir.f80>>>, !cir.ptr<!cir.long_double<!cir.f80>>
  // CIR: cir.store %[[C2]], %[[TMP3]] : !cir.long_double<!cir.f80>, !cir.ptr<!cir.long_double<!cir.f80>> tbaa(#tbaa[[LONG_DOUBLE]])
  // CIR: else
  // CIR: %[[C3:.*]] = cir.const #cir.fp<3
  // CIR: %[[TMP4:.*]] = cir.load deref %[[ARG_b]] : !cir.ptr<!cir.ptr<!cir.long_double<!cir.f80>>>, !cir.ptr<!cir.long_double<!cir.f80>>
  // CIR: cir.store %[[C3]], %[[TMP4]] : !cir.long_double<!cir.f80>, !cir.ptr<!cir.long_double<!cir.f80>> tbaa(#tbaa[[LONG_DOUBLE]])

  // LLVM-LABEL: void @test_long_long_and_long_double
  // LLVM: %[[ARG_a:.*]] = load i64, ptr %{{.*}}, align 8, !tbaa ![[TBAA_LONG_LONG:.*]]
  // LLVM: %[[COND:.*]] = icmp eq i64 %[[ARG_a]], 1
  // LLVM: %[[RET:.*]] = select i1 %[[COND]], x86_fp80 0xK40008000000000000000, x86_fp80 0xK4000C000000000000000
  // LLVM: store x86_fp80 %[[RET]], ptr %{{.*}}, align 16, !tbaa ![[TBAA_LONG_DOUBLE:.*]]
  // LLVM: ret void
  if (*a == 1L) {
    *b = 2.0L;
  } else {
    *b = 3.0L;
  }
}

void test_char(char *a, char* b) {
  // CIR-LABEL: cir.func @test_char
  // CIR: cir.scope
  // CIR: %[[TMP1:.*]] = cir.load deref %{{.*}} : !cir.ptr<!cir.ptr<!s8i>>, !cir.ptr<!s8i>
  // CIR: %[[TMP2:.*]] = cir.load %[[TMP1]] : !cir.ptr<!s8i>, !s8i tbaa(#tbaa[[CHAR]])
  // CIR: cir.if
  // CIR: %[[C2:.*]] = cir.const #cir.int<98> : !s32i
  // CIR: %[[C2_CHAR:.*]] = cir.cast(integral, %[[C2]] : !s32i), !s8i
  // CIR: %[[TMP3:.*]] = cir.load deref %[[ARG_b:.*]] : !cir.ptr<!cir.ptr<!s8i>>, !cir.ptr<!s8i>
  // CIR: cir.store %[[C2_CHAR]], %[[TMP3]] : !s8i, !cir.ptr<!s8i> tbaa(#tbaa[[CHAR]])
  // CIR: else
  // CIR: %[[C3:.*]] = cir.const #cir.int<0> : !s32i
  // CIR: %[[C3_CHAR:.*]] = cir.cast(integral, %[[C3]] : !s32i), !s8i
  // CIR: %[[TMP4:.*]] = cir.load deref %[[ARG_b]] : !cir.ptr<!cir.ptr<!s8i>>, !cir.ptr<!s8i>
  // CIR: cir.store %[[C3_CHAR]], %[[TMP4]] : !s8i, !cir.ptr<!s8i> tbaa(#tbaa[[CHAR]])


  // LLVM-LABEL: void @test_char
  // LLVM: %[[ARG_a:.*]] = load i8, ptr %{{.*}}, align 1, !tbaa ![[TBAA_CHAR:.*]]
  // LLVM: %[[COND:.*]] = icmp eq i8 %[[ARG_a]], 97
  // LLVM: %[[RET:.*]] = select i1 %[[COND]], i8 98, i8 0
  // LLVM: store i8 %[[RET]], ptr %{{.*}}, align 1, !tbaa ![[TBAA_CHAR]]
  // LLVM: ret void
  if (*a == 'a') {
    *b = 'b';
  }
  else {
    *b = '\0';
  }
}

// LLVM: ![[TBAA_INT]] = !{![[TBAA_INT_PARENT:.*]], ![[TBAA_INT_PARENT]], i64 0}
// LLVM: ![[TBAA_INT_PARENT]] = !{!"int", ![[CHAR:.*]], i64 0}
// LLVM: ![[CHAR]] = !{!"omnipotent char", ![[ROOT:.*]], i64 0}
// LLVM: ![[ROOT]] = !{!"Simple C/C++ TBAA"}
// LLVM: ![[TBAA_FLOAT]] = !{![[TBAA_FLOAT_PARENT:.*]], ![[TBAA_FLOAT_PARENT]], i64 0}
// LLVM: ![[TBAA_FLOAT_PARENT]] = !{!"float", ![[CHAR]], i64 0}
// LLVM: ![[TBAA_LONG]] = !{![[TBAA_LONG_PARENT:.*]], ![[TBAA_LONG_PARENT]], i64 0}
// LLVM: ![[TBAA_LONG_PARENT]] = !{!"long", ![[CHAR]], i64 0}
// LLVM: ![[TBAA_DOUBLE]] = !{![[TBAA_DOUBLE_PARENT:.*]], ![[TBAA_DOUBLE_PARENT]], i64 0}
// LLVM: ![[TBAA_DOUBLE_PARENT]] = !{!"double", ![[CHAR]], i64 0}
// LLVM: ![[TBAA_LONG_DOUBLE]] = !{![[TBAA_LONG_DOUBLE_PARENT:.*]], ![[TBAA_LONG_DOUBLE_PARENT]], i64 0}
// LLVM: ![[TBAA_LONG_DOUBLE_PARENT]] = !{!"long double", ![[CHAR]], i64 0}
// LLVM: ![[TBAA_CHAR]] = !{![[CHAR]], ![[CHAR]], i64 0}
