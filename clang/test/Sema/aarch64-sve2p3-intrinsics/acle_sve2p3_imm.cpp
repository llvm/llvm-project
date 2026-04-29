// REQUIRES: aarch64-registered-target

// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +sve2 -target-feature +sve2p3 -fsyntax-only -verify %s

#include <arm_sve.h>



svint8_t test_svqshrn_n_s8_s16_x2(svint16x2_t zn, uint64_t imm)
{
  svqshrn_n_s8_s16_x2(zn, 0); // expected-error-re {{argument value {{[0-9]+}} is outside the valid range [1, 8]}}
  svqshrn_n_s8_s16_x2(zn, 9); // expected-error-re {{argument value {{[0-9]+}} is outside the valid range [1, 8]}}
  svqshrn_n_s8_s16_x2(zn, -1); // expected-error-re {{argument value {{[0-9]+}} is outside the valid range [1, 8]}}

  svqshrn_n_s8_s16_x2(zn, imm); // expected-error-re {{argument to {{.+}} must be a constant integer}}}}
}

svint16_t test_svqshrn_n_s16_s32_x2(svint32x2_t zn, uint64_t imm)
{
  svqshrn_n_s16_s32_x2(zn, 0); // expected-error-re {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  svqshrn_n_s16_s32_x2(zn, 17); // expected-error-re {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  svqshrn_n_s16_s32_x2(zn, -1); // expected-error-re {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}

  svqshrn_n_s16_s32_x2(zn, imm); // expected-error-re {{argument to {{.+}} must be a constant integer}}}}
}

svuint8_t test_svqshrn_n_u8_u16_x2(svuint16x2_t zn, uint64_t imm)
{
  svqshrn_n_u8_u16_x2(zn, 0); // expected-error-re {{argument value {{[0-9]+}} is outside the valid range [1, 8]}}
  svqshrn_n_u8_u16_x2(zn, 9); // expected-error-re {{argument value {{[0-9]+}} is outside the valid range [1, 8]}}
  svqshrn_n_u8_u16_x2(zn, -1); // expected-error-re {{argument value {{[0-9]+}} is outside the valid range [1, 8]}}

  svqshrn_n_u8_u16_x2(zn, imm); // expected-error-re {{argument to {{.+}} must be a constant integer}}}}
}

svuint16_t test_svqshrn_n_u16_u32_x2(svuint32x2_t zn, uint64_t imm)
{
  svqshrn_n_u16_u32_x2(zn, 0); // expected-error-re {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  svqshrn_n_u16_u32_x2(zn, 17); // expected-error-re {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  svqshrn_n_u16_u32_x2(zn, -1); // expected-error-re {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}

  svqshrn_n_u16_u32_x2(zn, imm); // expected-error-re {{argument to {{.+}} must be a constant integer}}}}
}

svuint16_t test_svqshrun_n_u16_s32_x2(svint32x2_t zn, uint64_t imm)
{
  svqshrun_n_u16_s32_x2(zn, 0); // expected-error-re {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  svqshrun_n_u16_s32_x2(zn, 17); // expected-error-re {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  svqshrun_n_u16_s32_x2(zn, -1); // expected-error-re {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}

  svqshrun_n_u16_s32_x2(zn, imm); // expected-error-re {{argument to {{.+}} must be a constant integer}}}}
}

svuint8_t test_svqshrun_n_u8_s16_x2(svint16x2_t zn, uint64_t imm)
{
  svqshrun_n_u8_s16_x2(zn, 0); // expected-error-re {{argument value {{[0-9]+}} is outside the valid range [1, 8]}}
  svqshrun_n_u8_s16_x2(zn, 9); // expected-error-re {{argument value {{[0-9]+}} is outside the valid range [1, 8]}}
  svqshrun_n_u8_s16_x2(zn, -1); // expected-error-re {{argument value {{[0-9]+}} is outside the valid range [1, 8]}}

  svqshrun_n_u8_s16_x2(zn, imm); // expected-error-re {{argument to {{.+}} must be a constant integer}}}}
}

void test_svqrshrn_n_s8_s16_x2(svint16x2_t zn, uint64_t imm) {
  svqrshrn_n_s8_s16_x2(zn, 0); // expected-error-re {{argument value {{[0-9]+}} is outside the valid range [1, 8]}}
  svqrshrn_n_s8_s16_x2(zn, 9); // expected-error-re {{argument value {{[0-9]+}} is outside the valid range [1, 8]}}
  svqrshrn_n_s8_s16_x2(zn, -1); // expected-error-re {{argument value {{[0-9]+}} is outside the valid range [1, 8]}}

  svqrshrn_n_s8_s16_x2(zn, imm); // expected-error-re {{argument to {{.+}} must be a constant integer}}}}
}

svuint8_t test_svqrshrn_n_u8_u16_x2(svuint16x2_t zn, uint64_t imm)
{
  svqrshrn_n_u8_u16_x2(zn, 0); // expected-error-re {{argument value {{[0-9]+}} is outside the valid range [1, 8]}}
  svqrshrn_n_u8_u16_x2(zn, 9); // expected-error-re {{argument value {{[0-9]+}} is outside the valid range [1, 8]}}
  svqrshrn_n_u8_u16_x2(zn, -1); // expected-error-re {{argument value {{[0-9]+}} is outside the valid range [1, 8]}}

  svqrshrn_n_u8_u16_x2(zn, imm); // expected-error-re {{argument to {{.+}} must be a constant integer}}}}
}

svuint8_t test_svqrshrun_n_u8_s16_x2(svint16x2_t zn, uint64_t imm)
{
  svqrshrun_n_u8_s16_x2(zn, 0); // expected-error-re {{argument value {{[0-9]+}} is outside the valid range [1, 8]}}
  svqrshrun_n_u8_s16_x2(zn, 9); // expected-error-re {{argument value {{[0-9]+}} is outside the valid range [1, 8]}}
  svqrshrun_n_u8_s16_x2(zn, -1); // expected-error-re {{argument value {{[0-9]+}} is outside the valid range [1, 8]}}

  svqrshrun_n_u8_s16_x2(zn, imm); // expected-error-re {{argument to {{.+}} must be a constant integer}}}}
}
