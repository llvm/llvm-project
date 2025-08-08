// RUN: %clang_cc1 -triple aarch64-linux-gnu -ast-dump %s | FileCheck %s

int __attribute__((target_version("sve2-bitperm + sha2"))) foov(void) { return 1; }
int __attribute__((target_clones(" lse + fp + sha3 ", "default"))) fooc(void) { return 2; }

int __attribute__((target_version("aes;priority=1"))) explicit_priority(void) { return 1; }
int __attribute__((target_version("bf16;priority=2"))) explicit_priority(void) { return 2; }
int __attribute__((target_version("crc;priority=4"))) explicit_priority(void) { return 4; }
int __attribute__((target_version("dpb2;priority=8"))) explicit_priority(void) { return 8; }
int __attribute__((target_version("fp16fml;priority=16"))) explicit_priority(void) { return 16; }

int __attribute__((target_clones("simd;priority=31", "default"))) explicit_priority(void) {
  return 0;
}

// CHECK: TargetVersionAttr {{.*}} "sve2-bitperm+sha2"
// CHECK: TargetClonesAttr {{.*}} fp+lse+sha3 default

// CHECK: TargetVersionAttr {{.*}} "aes+P0"
// CHECK: TargetVersionAttr {{.*}} "bf16+P1"
// CHECK: TargetVersionAttr {{.*}} "crc+P2"
// CHECK: TargetVersionAttr {{.*}} "dpb2+P3"
// CHECK: TargetVersionAttr {{.*}} "fp16fml+P4"
// CHECK: TargetClonesAttr {{.*}} simd+P0+P1+P2+P3+P4 default
