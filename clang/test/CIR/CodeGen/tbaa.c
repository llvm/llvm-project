// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir -O1
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// CIR: #tbaa[[TBAA_NO:.*]] = #cir.tbaa
void f(int *a, float *b) {
  // CIR: cir.scope
  // CIR: %[[TMP1:.*]] = cir.load deref %{{.*}} : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i> tbaa([#tbaa[[TBAA_NO]]])
  // CIR: %[[TMP2:.*]] = cir.load %[[TMP1]] : !cir.ptr<!s32i>, !s32i tbaa([#tbaa[[TBAA_NO]]])
  // CIR: cir.if
  // CIR: %[[C2:.*]] = cir.const #cir.fp<2
  // CIR: %[[TMP3:.*]] = cir.load deref %[[ARG_b:.*]] : !cir.ptr<!cir.ptr<!cir.float>>, !cir.ptr<!cir.float> tbaa([#tbaa[[TBAA_NO]]])
  // CIR: cir.store %[[C2]], %[[TMP3]] : !cir.float, !cir.ptr<!cir.float> tbaa([#tbaa[[TBAA_NO]]])
  // CIR: else
  // CIR: %[[C3:.*]] = cir.const #cir.fp<3
  // CIR: %[[TMP4:.*]] = cir.load deref %[[ARG_b]] : !cir.ptr<!cir.ptr<!cir.float>>, !cir.ptr<!cir.float> tbaa([#tbaa[[TBAA_NO]]])
  // CIR: cir.store %[[C3]], %[[TMP4]] : !cir.float, !cir.ptr<!cir.float> tbaa([#tbaa[[TBAA_NO]]])
  if (*a == 1) {
    *b = 2.0f;
  } else {
    *b = 3.0f;
  }
}
