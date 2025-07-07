// RUN: %clang_cc1 -triple aarch64-linux-gnu -ast-dump %s | FileCheck %s

int __attribute__((target_version("sve2-bitperm + sha2"))) foov(void) { return 1; }
int __attribute__((target_clones(" lse + fp + sha3 ", "default"))) fooc(void) { return 2; }

int __attribute__((target_version("priority=1;aes"))) explicit_priority(void) { return 1; }
int __attribute__((target_version("priority=2;bf16"))) explicit_priority(void) { return 2; }
int __attribute__((target_version("priority=3;bti"))) explicit_priority(void) { return 3; }
int __attribute__((target_version("priority=4;crc"))) explicit_priority(void) { return 4; }
int __attribute__((target_version("priority=5;dit"))) explicit_priority(void) { return 5; }
int __attribute__((target_version("priority=6;dotprod"))) explicit_priority(void) { return 6; }
int __attribute__((target_version("priority=7;dpb"))) explicit_priority(void) { return 7; }
int __attribute__((target_version("priority=8;dpb2"))) explicit_priority(void) { return 8; }
int __attribute__((target_version("f32mm;priority=9"))) explicit_priority(void) { return 9; }
int __attribute__((target_version("f64mm;priority=10"))) explicit_priority(void) { return 10; }
int __attribute__((target_version("fcma;priority=11"))) explicit_priority(void) { return 11; }
int __attribute__((target_version("flagm;priority=12"))) explicit_priority(void) { return 12; }
int __attribute__((target_version("flagm2;priority=13"))) explicit_priority(void) { return 13; }
int __attribute__((target_version("fp;priority=14"))) explicit_priority(void) { return 14; }
int __attribute__((target_version("fp16;priority=15"))) explicit_priority(void) { return 15; }
int __attribute__((target_version("fp16fml;priority=16"))) explicit_priority(void) { return 16; }

int __attribute__((target_clones(
    "priority=17;frintts",
    "priority=18;i8mm",
    "priority=19;jscvt",
    "priority=20;lse",
    "priority=21;memtag",
    "priority=22;mops",
    "priority=23;rcpc",
    "priority=24;rcpc2",
    "rcpc3;priority=25",
    "rdm;priority=26",
    "rng;priority=27",
    "sb;priority=28",
    "sha2;priority=29",
    "sha3;priority=30",
    "simd;priority=31",
    "sm4;priority=32",
    "default"))) explicit_priority(void) { return 0; }

// CHECK: TargetVersionAttr {{.*}} "sve2-bitperm+sha2"
// CHECK: TargetClonesAttr {{.*}} fp+lse+sha3 default

// CHECK: TargetVersionAttr {{.*}} "priority1+priority2+priority3+priority4+priority5+aes"
// CHECK: TargetVersionAttr {{.*}} "priority1+priority2+priority3+priority4+bf16"
// CHECK: TargetVersionAttr {{.*}} "priority1+priority2+priority3+priority5+bti"
// CHECK: TargetVersionAttr {{.*}} "priority1+priority2+priority3+crc"
// CHECK: TargetVersionAttr {{.*}} "priority1+priority2+priority4+priority5+dit"
// CHECK: TargetVersionAttr {{.*}} "priority1+priority2+priority4+dotprod"
// CHECK: TargetVersionAttr {{.*}} "priority1+priority2+priority5+dpb"
// CHECK: TargetVersionAttr {{.*}} "priority1+priority2+dpb2"
// CHECK: TargetVersionAttr {{.*}} "f32mm+priority1+priority3+priority4+priority5"
// CHECK: TargetVersionAttr {{.*}} "f64mm+priority1+priority3+priority4"
// CHECK: TargetVersionAttr {{.*}} "fcma+priority1+priority3+priority5"
// CHECK: TargetVersionAttr {{.*}} "flagm+priority1+priority3"
// CHECK: TargetVersionAttr {{.*}} "flagm2+priority1+priority4+priority5"
// CHECK: TargetVersionAttr {{.*}} "fp+priority1+priority4"
// CHECK: TargetVersionAttr {{.*}} "fp16+priority1+priority5"
// CHECK: TargetVersionAttr {{.*}} "fp16fml+priority1"

// CHECK: TargetClonesAttr
// CHECK: priority2+priority3+priority4+priority5+frintts
// CHECK: priority2+priority3+priority4+i8mm
// CHECK: priority2+priority3+priority5+jscvt
// CHECK: priority2+priority3+lse
// CHECK: priority2+priority4+priority5+memtag
// CHECK: priority2+priority4+mops
// CHECK: priority2+priority5+rcpc
// CHECK: priority2+rcpc2
// CHECK: rcpc3+priority3+priority4+priority5
// CHECK: rdm+priority3+priority4
// CHECK: rng+priority3+priority5
// CHECK: sb+priority3
// CHECK: sha2+priority4+priority5
// CHECK: sha3+priority4
// CHECK: simd+priority5
// CHECK: sm4
// CHECK: default
