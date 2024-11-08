// RUN: %clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon \
// RUN:   -target-feature +sha2 -target-feature +aes \
// RUN:   -disable-O0-optnone -fclangir -emit-cir -o %t.cir %s 
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon \
// RUN:   -target-feature +sha2 -target-feature +aes \
// RUN:   -disable-O0-optnone  -emit-llvm -o - %s \
// RUN: | opt -S -passes=mem2reg,simplifycfg -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// REQUIRES: aarch64-registered-target || arm-registered-target

#include <arm_neon.h>

uint8x16_t test_vaesmcq_u8(uint8x16_t data) {
  return vaesmcq_u8(data);

  // CIR-LABEL: vaesmcq_u8
  // {{%.*}} = cir.llvm.intrinsic "aarch64.crypto.aesmc" {{%.*}} : (!cir.vector<!u8i x 16>) -> !cir.vector<!u8i x 16>

  // LLVM: {{.*}}vaesmcq_u8(<16 x i8>{{.*}}[[DATA:%.*]])
  // LLVM: [[RES:%.*]] = call <16 x i8> @llvm.aarch64.crypto.aesmc(<16 x i8> [[DATA]])
  // LLVM: ret <16 x i8> [[RES]]
}
