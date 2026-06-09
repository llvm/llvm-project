// RUN: %clang_cc1 -triple x86_64-unknown-unknown -std=c2y -isystem %S/../Sema/Inputs -emit-llvm -o - %s | FileCheck %s --check-prefix=LE
// RUN: %clang_cc1 -triple s390x-unknown-unknown  -std=c2y -isystem %S/../Sema/Inputs -emit-llvm -o - %s | FileCheck %s --check-prefix=BE

#include <stdbit.h>

// 8-bit: single byte load, no bswap on either target.
// LE-LABEL: test_leu8
// LE: load i8, ptr {{.*}}, align 1
// LE-NOT: bswap
// BE-LABEL: test_leu8
// BE: load i8, ptr {{.*}}, align 1
// BE-NOT: bswap
__UINT8_TYPE__ test_leu8(const unsigned char *p) { return stdc_load8_leu8(p); }

// LE load on LE target: no bswap. On BE target: bswap needed.
// LE-LABEL: test_leu16
// LE: load i16, ptr {{.*}}, align 1
// LE-NOT: bswap
// BE-LABEL: test_leu16
// BE: load i16, ptr {{.*}}, align 1
// BE: call i16 @llvm.bswap.i16
__UINT16_TYPE__ test_leu16(const unsigned char *p) { return stdc_load8_leu16(p); }

// LE-LABEL: test_leu32
// LE: load i32, ptr {{.*}}, align 1
// LE-NOT: bswap
// BE-LABEL: test_leu32
// BE: load i32, ptr {{.*}}, align 1
// BE: call i32 @llvm.bswap.i32
__UINT32_TYPE__ test_leu32(const unsigned char *p) { return stdc_load8_leu32(p); }

// LE-LABEL: test_leu64
// LE: load i64, ptr {{.*}}, align 1
// LE-NOT: bswap
// BE-LABEL: test_leu64
// BE: load i64, ptr {{.*}}, align 1
// BE: call i64 @llvm.bswap.i64
__UINT64_TYPE__ test_leu64(const unsigned char *p) { return stdc_load8_leu64(p); }

// BE load on LE target: bswap needed. On BE target: no bswap.
// LE-LABEL: test_beu16
// LE: load i16, ptr {{.*}}, align 1
// LE: call i16 @llvm.bswap.i16
// BE-LABEL: test_beu16
// BE: load i16, ptr {{.*}}, align 1
// BE-NOT: bswap
__UINT16_TYPE__ test_beu16(const unsigned char *p) { return stdc_load8_beu16(p); }

// LE-LABEL: test_beu32
// LE: load i32, ptr {{.*}}, align 1
// LE: call i32 @llvm.bswap.i32
// BE-LABEL: test_beu32
// BE: load i32, ptr {{.*}}, align 1
// BE-NOT: bswap
__UINT32_TYPE__ test_beu32(const unsigned char *p) { return stdc_load8_beu32(p); }

// LE-LABEL: test_beu64
// LE: load i64, ptr {{.*}}, align 1
// LE: call i64 @llvm.bswap.i64
// BE-LABEL: test_beu64
// BE: load i64, ptr {{.*}}, align 1
// BE-NOT: bswap
__UINT64_TYPE__ test_beu64(const unsigned char *p) { return stdc_load8_beu64(p); }

// Aligned variants use natural alignment instead of align 1.
// LE-LABEL: test_aligned_leu32
// LE: load i32, ptr {{.*}}, align 4
// LE-NOT: bswap
// BE-LABEL: test_aligned_leu32
// BE: load i32, ptr {{.*}}, align 4
// BE: call i32 @llvm.bswap.i32
__UINT32_TYPE__ test_aligned_leu32(const unsigned char *p) { return stdc_load8_aligned_leu32(p); }

// LE-LABEL: test_aligned_beu32
// LE: load i32, ptr {{.*}}, align 4
// LE: call i32 @llvm.bswap.i32
// BE-LABEL: test_aligned_beu32
// BE: load i32, ptr {{.*}}, align 4
// BE-NOT: bswap
__UINT32_TYPE__ test_aligned_beu32(const unsigned char *p) { return stdc_load8_aligned_beu32(p); }

// Signed variants: same load+bswap logic, result type is signed.
// LE-LABEL: test_les32
// LE: load i32, ptr {{.*}}, align 1
// LE-NOT: bswap
// BE-LABEL: test_les32
// BE: load i32, ptr {{.*}}, align 1
// BE: call i32 @llvm.bswap.i32
__INT32_TYPE__ test_les32(const unsigned char *p) { return stdc_load8_les32(p); }

// LE-LABEL: test_bes32
// LE: load i32, ptr {{.*}}, align 1
// LE: call i32 @llvm.bswap.i32
// BE-LABEL: test_bes32
// BE: load i32, ptr {{.*}}, align 1
// BE-NOT: bswap
__INT32_TYPE__ test_bes32(const unsigned char *p) { return stdc_load8_bes32(p); }
