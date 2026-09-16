// RUN: %clang_cc1 -triple x86_64-unknown-unknown -std=c2y -isystem %S/../Sema/Inputs -emit-llvm -o - %s | FileCheck %s --check-prefix=LE
// RUN: %clang_cc1 -triple s390x-unknown-unknown  -std=c2y -isystem %S/../Sema/Inputs -emit-llvm -o - %s | FileCheck %s --check-prefix=BE

#include <stdbit.h>

// 8-bit: single byte load, no bswap on either target.
// LE-LABEL: @test_leu8(
// LE: load i8, ptr {{.+}}, align 1
// LE-NOT: bswap
// BE-LABEL: @test_leu8(
// BE: load i8, ptr {{.+}}, align 1
// BE-NOT: bswap
__UINT_LEAST8_TYPE__ test_leu8(const unsigned char *p) { return stdc_load8_leu8(p); }

// LE load on LE target: no bswap. On BE target: bswap needed.
// LE-LABEL: @test_leu16(
// LE: load i16, ptr {{.+}}, align 1
// LE-NOT: bswap
// BE-LABEL: @test_leu16(
// BE: load i16, ptr {{.+}}, align 1
// BE: call i16 @llvm.bswap.i16(
__UINT_LEAST16_TYPE__ test_leu16(const unsigned char *p) { return stdc_load8_leu16(p); }

// LE-LABEL: @test_leu32(
// LE: load i32, ptr {{.+}}, align 1
// LE-NOT: bswap
// BE-LABEL: @test_leu32(
// BE: load i32, ptr {{.+}}, align 1
// BE: call i32 @llvm.bswap.i32(
__UINT_LEAST32_TYPE__ test_leu32(const unsigned char *p) { return stdc_load8_leu32(p); }

// LE-LABEL: @test_leu64(
// LE: load i64, ptr {{.+}}, align 1
// LE-NOT: bswap
// BE-LABEL: @test_leu64(
// BE: load i64, ptr {{.+}}, align 1
// BE: call i64 @llvm.bswap.i64(
__UINT_LEAST64_TYPE__ test_leu64(const unsigned char *p) { return stdc_load8_leu64(p); }

// BE load on LE target: bswap needed. On BE target: no bswap.
// LE-LABEL: @test_beu16(
// LE: load i16, ptr {{.+}}, align 1
// LE: call i16 @llvm.bswap.i16(
// BE-LABEL: @test_beu16(
// BE: load i16, ptr {{.+}}, align 1
// BE-NOT: bswap
__UINT_LEAST16_TYPE__ test_beu16(const unsigned char *p) { return stdc_load8_beu16(p); }

// LE-LABEL: @test_beu32(
// LE: load i32, ptr {{.+}}, align 1
// LE: call i32 @llvm.bswap.i32(
// BE-LABEL: @test_beu32(
// BE: load i32, ptr {{.+}}, align 1
// BE-NOT: bswap
__UINT_LEAST32_TYPE__ test_beu32(const unsigned char *p) { return stdc_load8_beu32(p); }

// LE-LABEL: @test_beu64(
// LE: load i64, ptr {{.+}}, align 1
// LE: call i64 @llvm.bswap.i64(
// BE-LABEL: @test_beu64(
// BE: load i64, ptr {{.+}}, align 1
// BE-NOT: bswap
__UINT_LEAST64_TYPE__ test_beu64(const unsigned char *p) { return stdc_load8_beu64(p); }

// Aligned variants use natural alignment instead of align 1.
// LE-LABEL: @test_aligned_leu32(
// LE: load i32, ptr {{.+}}, align 4
// LE-NOT: bswap
// BE-LABEL: @test_aligned_leu32(
// BE: load i32, ptr {{.+}}, align 4
// BE: call i32 @llvm.bswap.i32(
__UINT_LEAST32_TYPE__ test_aligned_leu32(const unsigned char *p) { return stdc_load8_aligned_leu32(p); }

// LE-LABEL: @test_aligned_beu32(
// LE: load i32, ptr {{.+}}, align 4
// LE: call i32 @llvm.bswap.i32(
// BE-LABEL: @test_aligned_beu32(
// BE: load i32, ptr {{.+}}, align 4
// BE-NOT: bswap
__UINT_LEAST32_TYPE__ test_aligned_beu32(const unsigned char *p) { return stdc_load8_aligned_beu32(p); }

// Signed variants: same load+bswap logic, result type is signed.
// LE-LABEL: @test_les32(
// LE: load i32, ptr {{.+}}, align 1
// LE-NOT: bswap
// BE-LABEL: @test_les32(
// BE: load i32, ptr {{.+}}, align 1
// BE: call i32 @llvm.bswap.i32(
__INT_LEAST32_TYPE__ test_les32(const unsigned char *p) { return stdc_load8_les32(p); }

// LE-LABEL: @test_bes32(
// LE: load i32, ptr {{.+}}, align 1
// LE: call i32 @llvm.bswap.i32(
// BE-LABEL: @test_bes32(
// BE: load i32, ptr {{.+}}, align 1
// BE-NOT: bswap
__INT_LEAST32_TYPE__ test_bes32(const unsigned char *p) { return stdc_load8_bes32(p); }

// Aligned load's alignment scales with the result type's width.
// LE-LABEL: @test_aligned_leu16(
// LE: load i16, ptr {{.+}}, align 2
// LE-NOT: bswap
// BE-LABEL: @test_aligned_leu16(
// BE: load i16, ptr {{.+}}, align 2
// BE: call i16 @llvm.bswap.i16(
__UINT_LEAST16_TYPE__ test_aligned_leu16(const unsigned char *p) { return stdc_load8_aligned_leu16(p); }

// LE-LABEL: @test_aligned_beu16(
// LE: load i16, ptr {{.+}}, align 2
// LE: call i16 @llvm.bswap.i16(
// BE-LABEL: @test_aligned_beu16(
// BE: load i16, ptr {{.+}}, align 2
// BE-NOT: bswap
__UINT_LEAST16_TYPE__ test_aligned_beu16(const unsigned char *p) { return stdc_load8_aligned_beu16(p); }

// LE-LABEL: @test_aligned_leu64(
// LE: load i64, ptr {{.+}}, align 8
// LE-NOT: bswap
// BE-LABEL: @test_aligned_leu64(
// BE: load i64, ptr {{.+}}, align 8
// BE: call i64 @llvm.bswap.i64(
__UINT_LEAST64_TYPE__ test_aligned_leu64(const unsigned char *p) { return stdc_load8_aligned_leu64(p); }

// LE-LABEL: @test_aligned_beu64(
// LE: load i64, ptr {{.+}}, align 8
// LE: call i64 @llvm.bswap.i64(
// BE-LABEL: @test_aligned_beu64(
// BE: load i64, ptr {{.+}}, align 8
// BE-NOT: bswap
__UINT_LEAST64_TYPE__ test_aligned_beu64(const unsigned char *p) { return stdc_load8_aligned_beu64(p); }

// Signed aligned variants use the same alignment lowering as unsigned.
// LE-LABEL: @test_aligned_les32(
// LE: load i32, ptr {{.+}}, align 4
// LE-NOT: bswap
// BE-LABEL: @test_aligned_les32(
// BE: load i32, ptr {{.+}}, align 4
// BE: call i32 @llvm.bswap.i32(
__INT_LEAST32_TYPE__ test_aligned_les32(const unsigned char *p) { return stdc_load8_aligned_les32(p); }

// LE-LABEL: @test_aligned_bes32(
// LE: load i32, ptr {{.+}}, align 4
// LE: call i32 @llvm.bswap.i32(
// BE-LABEL: @test_aligned_bes32(
// BE: load i32, ptr {{.+}}, align 4
// BE-NOT: bswap
__INT_LEAST32_TYPE__ test_aligned_bes32(const unsigned char *p) { return stdc_load8_aligned_bes32(p); }

// buf's alignment (4) must be kept, not forced down to align 1.
alignas(4) unsigned char buf4[4];
// LE-LABEL: @test_leu32_known_alignment(
// LE: load i32, ptr @buf4, align 4
// LE-NOT: bswap
// BE-LABEL: @test_leu32_known_alignment(
// BE: load i32, ptr @buf4, align 4
// BE: call i32 @llvm.bswap.i32(
__UINT_LEAST32_TYPE__ test_leu32_known_alignment(void) { return stdc_load8_leu32(buf4); }

// 8-bit unsigned big-endian, and the aligned variants of both endian orders.
// LE-LABEL: @test_beu8(
// LE: load i8, ptr {{.+}}, align 1
// LE-NOT: bswap
// BE-LABEL: @test_beu8(
// BE: load i8, ptr {{.+}}, align 1
// BE-NOT: bswap
__UINT_LEAST8_TYPE__ test_beu8(const unsigned char *p) { return stdc_load8_beu8(p); }

// LE-LABEL: @test_aligned_leu8(
// LE: load i8, ptr {{.+}}, align 1
// LE-NOT: bswap
// BE-LABEL: @test_aligned_leu8(
// BE: load i8, ptr {{.+}}, align 1
// BE-NOT: bswap
__UINT_LEAST8_TYPE__ test_aligned_leu8(const unsigned char *p) { return stdc_load8_aligned_leu8(p); }

// LE-LABEL: @test_aligned_beu8(
// LE: load i8, ptr {{.+}}, align 1
// LE-NOT: bswap
// BE-LABEL: @test_aligned_beu8(
// BE: load i8, ptr {{.+}}, align 1
// BE-NOT: bswap
__UINT_LEAST8_TYPE__ test_aligned_beu8(const unsigned char *p) { return stdc_load8_aligned_beu8(p); }

// Signed 8-bit variants (unaligned and aligned, both endian orders).
// LE-LABEL: @test_les8(
// LE: load i8, ptr {{.+}}, align 1
// LE-NOT: bswap
// BE-LABEL: @test_les8(
// BE: load i8, ptr {{.+}}, align 1
// BE-NOT: bswap
__INT_LEAST8_TYPE__ test_les8(const unsigned char *p) { return stdc_load8_les8(p); }

// LE-LABEL: @test_bes8(
// LE: load i8, ptr {{.+}}, align 1
// LE-NOT: bswap
// BE-LABEL: @test_bes8(
// BE: load i8, ptr {{.+}}, align 1
// BE-NOT: bswap
__INT_LEAST8_TYPE__ test_bes8(const unsigned char *p) { return stdc_load8_bes8(p); }

// LE-LABEL: @test_aligned_les8(
// LE: load i8, ptr {{.+}}, align 1
// LE-NOT: bswap
// BE-LABEL: @test_aligned_les8(
// BE: load i8, ptr {{.+}}, align 1
// BE-NOT: bswap
__INT_LEAST8_TYPE__ test_aligned_les8(const unsigned char *p) { return stdc_load8_aligned_les8(p); }

// LE-LABEL: @test_aligned_bes8(
// LE: load i8, ptr {{.+}}, align 1
// LE-NOT: bswap
// BE-LABEL: @test_aligned_bes8(
// BE: load i8, ptr {{.+}}, align 1
// BE-NOT: bswap
__INT_LEAST8_TYPE__ test_aligned_bes8(const unsigned char *p) { return stdc_load8_aligned_bes8(p); }

// Signed 16-bit, unaligned and aligned, both endian orders.
// LE-LABEL: @test_les16(
// LE: load i16, ptr {{.+}}, align 1
// LE-NOT: bswap
// BE-LABEL: @test_les16(
// BE: load i16, ptr {{.+}}, align 1
// BE: call i16 @llvm.bswap.i16(
__INT_LEAST16_TYPE__ test_les16(const unsigned char *p) { return stdc_load8_les16(p); }

// LE-LABEL: @test_bes16(
// LE: load i16, ptr {{.+}}, align 1
// LE: call i16 @llvm.bswap.i16(
// BE-LABEL: @test_bes16(
// BE: load i16, ptr {{.+}}, align 1
// BE-NOT: bswap
__INT_LEAST16_TYPE__ test_bes16(const unsigned char *p) { return stdc_load8_bes16(p); }

// LE-LABEL: @test_aligned_les16(
// LE: load i16, ptr {{.+}}, align 2
// LE-NOT: bswap
// BE-LABEL: @test_aligned_les16(
// BE: load i16, ptr {{.+}}, align 2
// BE: call i16 @llvm.bswap.i16(
__INT_LEAST16_TYPE__ test_aligned_les16(const unsigned char *p) { return stdc_load8_aligned_les16(p); }

// LE-LABEL: @test_aligned_bes16(
// LE: load i16, ptr {{.+}}, align 2
// LE: call i16 @llvm.bswap.i16(
// BE-LABEL: @test_aligned_bes16(
// BE: load i16, ptr {{.+}}, align 2
// BE-NOT: bswap
__INT_LEAST16_TYPE__ test_aligned_bes16(const unsigned char *p) { return stdc_load8_aligned_bes16(p); }

// Signed 64-bit, unaligned and aligned, both endian orders.
// LE-LABEL: @test_les64(
// LE: load i64, ptr {{.+}}, align 1
// LE-NOT: bswap
// BE-LABEL: @test_les64(
// BE: load i64, ptr {{.+}}, align 1
// BE: call i64 @llvm.bswap.i64(
__INT_LEAST64_TYPE__ test_les64(const unsigned char *p) { return stdc_load8_les64(p); }

// LE-LABEL: @test_bes64(
// LE: load i64, ptr {{.+}}, align 1
// LE: call i64 @llvm.bswap.i64(
// BE-LABEL: @test_bes64(
// BE: load i64, ptr {{.+}}, align 1
// BE-NOT: bswap
__INT_LEAST64_TYPE__ test_bes64(const unsigned char *p) { return stdc_load8_bes64(p); }

// LE-LABEL: @test_aligned_les64(
// LE: load i64, ptr {{.+}}, align 8
// LE-NOT: bswap
// BE-LABEL: @test_aligned_les64(
// BE: load i64, ptr {{.+}}, align 8
// BE: call i64 @llvm.bswap.i64(
__INT_LEAST64_TYPE__ test_aligned_les64(const unsigned char *p) { return stdc_load8_aligned_les64(p); }

// LE-LABEL: @test_aligned_bes64(
// LE: load i64, ptr {{.+}}, align 8
// LE: call i64 @llvm.bswap.i64(
// BE-LABEL: @test_aligned_bes64(
// BE: load i64, ptr {{.+}}, align 8
// BE-NOT: bswap
__INT_LEAST64_TYPE__ test_aligned_bes64(const unsigned char *p) { return stdc_load8_aligned_bes64(p); }

// buf16's alignment (16) exceeds the 4-byte minimum aligned_leu32 needs;
// the load should keep align 16.
alignas(16) unsigned char buf16[4];
// LE-LABEL: @test_aligned_leu32_stronger_known_alignment(
// LE: load i32, ptr @buf16, align 16
// LE-NOT: bswap
// BE-LABEL: @test_aligned_leu32_stronger_known_alignment(
// BE: load i32, ptr @buf16, align 16
// BE: call i32 @llvm.bswap.i32(
__UINT_LEAST32_TYPE__ test_aligned_leu32_stronger_known_alignment(void) { return stdc_load8_aligned_leu32(buf16); }
