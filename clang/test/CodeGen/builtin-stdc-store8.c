// RUN: %clang_cc1 -triple x86_64-unknown-unknown -std=c2y -isystem %S/../Sema/Inputs -O1 -emit-llvm -o - %s | FileCheck %s --check-prefix=LE
// RUN: %clang_cc1 -triple s390x-unknown-unknown  -std=c2y -isystem %S/../Sema/Inputs -O1 -emit-llvm -o - %s | FileCheck %s --check-prefix=BE

#include <stdbit.h>

// Every stdc_store8_* variant is lowered to an optional llvm.bswap (when the
// requested byte order differs from the target's) followed by one integer
// store. The unaligned variants store with align 1; the aligned_* variants use
// the natural alignment of the value's type. 8-bit stores never need a swap.

// LE-LABEL: @test_leu8(
// LE:      store i8 %v, ptr %p, align 1{{$}}
// LE-NEXT: ret void
// BE-LABEL: @test_leu8(
// BE:      store i8 %v, ptr %p, align 1{{$}}
// BE-NEXT: ret void
void test_leu8(__UINT_LEAST8_TYPE__ v, unsigned char *p) { stdc_store8_leu8(v, p); }

// LE-LABEL: @test_leu16(
// LE:      store i16 %v, ptr %p, align 1{{$}}
// LE-NEXT: ret void
// BE-LABEL: @test_leu16(
// BE:      %0 = tail call i16 @llvm.bswap.i16(i16 %v)
// BE-NEXT: store i16 %0, ptr %p, align 1{{$}}
// BE-NEXT: ret void
void test_leu16(__UINT_LEAST16_TYPE__ v, unsigned char *p) { stdc_store8_leu16(v, p); }

// LE-LABEL: @test_leu32(
// LE:      store i32 %v, ptr %p, align 1{{$}}
// LE-NEXT: ret void
// BE-LABEL: @test_leu32(
// BE:      %0 = tail call i32 @llvm.bswap.i32(i32 %v)
// BE-NEXT: store i32 %0, ptr %p, align 1{{$}}
// BE-NEXT: ret void
void test_leu32(__UINT_LEAST32_TYPE__ v, unsigned char *p) { stdc_store8_leu32(v, p); }

// LE-LABEL: @test_leu64(
// LE:      store i64 %v, ptr %p, align 1{{$}}
// LE-NEXT: ret void
// BE-LABEL: @test_leu64(
// BE:      %0 = tail call i64 @llvm.bswap.i64(i64 %v)
// BE-NEXT: store i64 %0, ptr %p, align 1{{$}}
// BE-NEXT: ret void
void test_leu64(__UINT_LEAST64_TYPE__ v, unsigned char *p) { stdc_store8_leu64(v, p); }

// LE-LABEL: @test_beu8(
// LE:      store i8 %v, ptr %p, align 1{{$}}
// LE-NEXT: ret void
// BE-LABEL: @test_beu8(
// BE:      store i8 %v, ptr %p, align 1{{$}}
// BE-NEXT: ret void
void test_beu8(__UINT_LEAST8_TYPE__ v, unsigned char *p) { stdc_store8_beu8(v, p); }

// LE-LABEL: @test_beu16(
// LE:      %0 = tail call i16 @llvm.bswap.i16(i16 %v)
// LE-NEXT: store i16 %0, ptr %p, align 1{{$}}
// LE-NEXT: ret void
// BE-LABEL: @test_beu16(
// BE:      store i16 %v, ptr %p, align 1{{$}}
// BE-NEXT: ret void
void test_beu16(__UINT_LEAST16_TYPE__ v, unsigned char *p) { stdc_store8_beu16(v, p); }

// LE-LABEL: @test_beu32(
// LE:      %0 = tail call i32 @llvm.bswap.i32(i32 %v)
// LE-NEXT: store i32 %0, ptr %p, align 1{{$}}
// LE-NEXT: ret void
// BE-LABEL: @test_beu32(
// BE:      store i32 %v, ptr %p, align 1{{$}}
// BE-NEXT: ret void
void test_beu32(__UINT_LEAST32_TYPE__ v, unsigned char *p) { stdc_store8_beu32(v, p); }

// LE-LABEL: @test_beu64(
// LE:      %0 = tail call i64 @llvm.bswap.i64(i64 %v)
// LE-NEXT: store i64 %0, ptr %p, align 1{{$}}
// LE-NEXT: ret void
// BE-LABEL: @test_beu64(
// BE:      store i64 %v, ptr %p, align 1{{$}}
// BE-NEXT: ret void
void test_beu64(__UINT_LEAST64_TYPE__ v, unsigned char *p) { stdc_store8_beu64(v, p); }

// LE-LABEL: @test_aligned_leu8(
// LE:      store i8 %v, ptr %p, align 1{{$}}
// LE-NEXT: ret void
// BE-LABEL: @test_aligned_leu8(
// BE:      store i8 %v, ptr %p, align 1{{$}}
// BE-NEXT: ret void
void test_aligned_leu8(__UINT_LEAST8_TYPE__ v, unsigned char *p) { stdc_store8_aligned_leu8(v, p); }

// LE-LABEL: @test_aligned_leu16(
// LE:      store i16 %v, ptr %p, align 2
// LE-NEXT: ret void
// BE-LABEL: @test_aligned_leu16(
// BE:      %0 = tail call i16 @llvm.bswap.i16(i16 %v)
// BE-NEXT: store i16 %0, ptr %p, align 2
// BE-NEXT: ret void
void test_aligned_leu16(__UINT_LEAST16_TYPE__ v, unsigned char *p) { stdc_store8_aligned_leu16(v, p); }

// LE-LABEL: @test_aligned_leu32(
// LE:      store i32 %v, ptr %p, align 4
// LE-NEXT: ret void
// BE-LABEL: @test_aligned_leu32(
// BE:      %0 = tail call i32 @llvm.bswap.i32(i32 %v)
// BE-NEXT: store i32 %0, ptr %p, align 4
// BE-NEXT: ret void
void test_aligned_leu32(__UINT_LEAST32_TYPE__ v, unsigned char *p) { stdc_store8_aligned_leu32(v, p); }

// LE-LABEL: @test_aligned_leu64(
// LE:      store i64 %v, ptr %p, align 8
// LE-NEXT: ret void
// BE-LABEL: @test_aligned_leu64(
// BE:      %0 = tail call i64 @llvm.bswap.i64(i64 %v)
// BE-NEXT: store i64 %0, ptr %p, align 8
// BE-NEXT: ret void
void test_aligned_leu64(__UINT_LEAST64_TYPE__ v, unsigned char *p) { stdc_store8_aligned_leu64(v, p); }

// LE-LABEL: @test_aligned_beu8(
// LE:      store i8 %v, ptr %p, align 1{{$}}
// LE-NEXT: ret void
// BE-LABEL: @test_aligned_beu8(
// BE:      store i8 %v, ptr %p, align 1{{$}}
// BE-NEXT: ret void
void test_aligned_beu8(__UINT_LEAST8_TYPE__ v, unsigned char *p) { stdc_store8_aligned_beu8(v, p); }

// LE-LABEL: @test_aligned_beu16(
// LE:      %0 = tail call i16 @llvm.bswap.i16(i16 %v)
// LE-NEXT: store i16 %0, ptr %p, align 2
// LE-NEXT: ret void
// BE-LABEL: @test_aligned_beu16(
// BE:      store i16 %v, ptr %p, align 2
// BE-NEXT: ret void
void test_aligned_beu16(__UINT_LEAST16_TYPE__ v, unsigned char *p) { stdc_store8_aligned_beu16(v, p); }

// LE-LABEL: @test_aligned_beu32(
// LE:      %0 = tail call i32 @llvm.bswap.i32(i32 %v)
// LE-NEXT: store i32 %0, ptr %p, align 4
// LE-NEXT: ret void
// BE-LABEL: @test_aligned_beu32(
// BE:      store i32 %v, ptr %p, align 4
// BE-NEXT: ret void
void test_aligned_beu32(__UINT_LEAST32_TYPE__ v, unsigned char *p) { stdc_store8_aligned_beu32(v, p); }

// LE-LABEL: @test_aligned_beu64(
// LE:      %0 = tail call i64 @llvm.bswap.i64(i64 %v)
// LE-NEXT: store i64 %0, ptr %p, align 8
// LE-NEXT: ret void
// BE-LABEL: @test_aligned_beu64(
// BE:      store i64 %v, ptr %p, align 8
// BE-NEXT: ret void
void test_aligned_beu64(__UINT_LEAST64_TYPE__ v, unsigned char *p) { stdc_store8_aligned_beu64(v, p); }

// LE-LABEL: @test_les8(
// LE:      store i8 %v, ptr %p, align 1{{$}}
// LE-NEXT: ret void
// BE-LABEL: @test_les8(
// BE:      store i8 %v, ptr %p, align 1{{$}}
// BE-NEXT: ret void
void test_les8(__INT_LEAST8_TYPE__ v, unsigned char *p) { stdc_store8_les8(v, p); }

// LE-LABEL: @test_les16(
// LE:      store i16 %v, ptr %p, align 1{{$}}
// LE-NEXT: ret void
// BE-LABEL: @test_les16(
// BE:      %0 = tail call i16 @llvm.bswap.i16(i16 %v)
// BE-NEXT: store i16 %0, ptr %p, align 1{{$}}
// BE-NEXT: ret void
void test_les16(__INT_LEAST16_TYPE__ v, unsigned char *p) { stdc_store8_les16(v, p); }

// LE-LABEL: @test_les32(
// LE:      store i32 %v, ptr %p, align 1{{$}}
// LE-NEXT: ret void
// BE-LABEL: @test_les32(
// BE:      %0 = tail call i32 @llvm.bswap.i32(i32 %v)
// BE-NEXT: store i32 %0, ptr %p, align 1{{$}}
// BE-NEXT: ret void
void test_les32(__INT_LEAST32_TYPE__ v, unsigned char *p) { stdc_store8_les32(v, p); }

// LE-LABEL: @test_les64(
// LE:      store i64 %v, ptr %p, align 1{{$}}
// LE-NEXT: ret void
// BE-LABEL: @test_les64(
// BE:      %0 = tail call i64 @llvm.bswap.i64(i64 %v)
// BE-NEXT: store i64 %0, ptr %p, align 1{{$}}
// BE-NEXT: ret void
void test_les64(__INT_LEAST64_TYPE__ v, unsigned char *p) { stdc_store8_les64(v, p); }

// LE-LABEL: @test_bes8(
// LE:      store i8 %v, ptr %p, align 1{{$}}
// LE-NEXT: ret void
// BE-LABEL: @test_bes8(
// BE:      store i8 %v, ptr %p, align 1{{$}}
// BE-NEXT: ret void
void test_bes8(__INT_LEAST8_TYPE__ v, unsigned char *p) { stdc_store8_bes8(v, p); }

// LE-LABEL: @test_bes16(
// LE:      %0 = tail call i16 @llvm.bswap.i16(i16 %v)
// LE-NEXT: store i16 %0, ptr %p, align 1{{$}}
// LE-NEXT: ret void
// BE-LABEL: @test_bes16(
// BE:      store i16 %v, ptr %p, align 1{{$}}
// BE-NEXT: ret void
void test_bes16(__INT_LEAST16_TYPE__ v, unsigned char *p) { stdc_store8_bes16(v, p); }

// LE-LABEL: @test_bes32(
// LE:      %0 = tail call i32 @llvm.bswap.i32(i32 %v)
// LE-NEXT: store i32 %0, ptr %p, align 1{{$}}
// LE-NEXT: ret void
// BE-LABEL: @test_bes32(
// BE:      store i32 %v, ptr %p, align 1{{$}}
// BE-NEXT: ret void
void test_bes32(__INT_LEAST32_TYPE__ v, unsigned char *p) { stdc_store8_bes32(v, p); }

// LE-LABEL: @test_bes64(
// LE:      %0 = tail call i64 @llvm.bswap.i64(i64 %v)
// LE-NEXT: store i64 %0, ptr %p, align 1{{$}}
// LE-NEXT: ret void
// BE-LABEL: @test_bes64(
// BE:      store i64 %v, ptr %p, align 1{{$}}
// BE-NEXT: ret void
void test_bes64(__INT_LEAST64_TYPE__ v, unsigned char *p) { stdc_store8_bes64(v, p); }

// LE-LABEL: @test_aligned_les8(
// LE:      store i8 %v, ptr %p, align 1{{$}}
// LE-NEXT: ret void
// BE-LABEL: @test_aligned_les8(
// BE:      store i8 %v, ptr %p, align 1{{$}}
// BE-NEXT: ret void
void test_aligned_les8(__INT_LEAST8_TYPE__ v, unsigned char *p) { stdc_store8_aligned_les8(v, p); }

// LE-LABEL: @test_aligned_les16(
// LE:      store i16 %v, ptr %p, align 2
// LE-NEXT: ret void
// BE-LABEL: @test_aligned_les16(
// BE:      %0 = tail call i16 @llvm.bswap.i16(i16 %v)
// BE-NEXT: store i16 %0, ptr %p, align 2
// BE-NEXT: ret void
void test_aligned_les16(__INT_LEAST16_TYPE__ v, unsigned char *p) { stdc_store8_aligned_les16(v, p); }

// LE-LABEL: @test_aligned_les32(
// LE:      store i32 %v, ptr %p, align 4
// LE-NEXT: ret void
// BE-LABEL: @test_aligned_les32(
// BE:      %0 = tail call i32 @llvm.bswap.i32(i32 %v)
// BE-NEXT: store i32 %0, ptr %p, align 4
// BE-NEXT: ret void
void test_aligned_les32(__INT_LEAST32_TYPE__ v, unsigned char *p) { stdc_store8_aligned_les32(v, p); }

// LE-LABEL: @test_aligned_les64(
// LE:      store i64 %v, ptr %p, align 8
// LE-NEXT: ret void
// BE-LABEL: @test_aligned_les64(
// BE:      %0 = tail call i64 @llvm.bswap.i64(i64 %v)
// BE-NEXT: store i64 %0, ptr %p, align 8
// BE-NEXT: ret void
void test_aligned_les64(__INT_LEAST64_TYPE__ v, unsigned char *p) { stdc_store8_aligned_les64(v, p); }

// LE-LABEL: @test_aligned_bes8(
// LE:      store i8 %v, ptr %p, align 1{{$}}
// LE-NEXT: ret void
// BE-LABEL: @test_aligned_bes8(
// BE:      store i8 %v, ptr %p, align 1{{$}}
// BE-NEXT: ret void
void test_aligned_bes8(__INT_LEAST8_TYPE__ v, unsigned char *p) { stdc_store8_aligned_bes8(v, p); }

// LE-LABEL: @test_aligned_bes16(
// LE:      %0 = tail call i16 @llvm.bswap.i16(i16 %v)
// LE-NEXT: store i16 %0, ptr %p, align 2
// LE-NEXT: ret void
// BE-LABEL: @test_aligned_bes16(
// BE:      store i16 %v, ptr %p, align 2
// BE-NEXT: ret void
void test_aligned_bes16(__INT_LEAST16_TYPE__ v, unsigned char *p) { stdc_store8_aligned_bes16(v, p); }

// LE-LABEL: @test_aligned_bes32(
// LE:      %0 = tail call i32 @llvm.bswap.i32(i32 %v)
// LE-NEXT: store i32 %0, ptr %p, align 4
// LE-NEXT: ret void
// BE-LABEL: @test_aligned_bes32(
// BE:      store i32 %v, ptr %p, align 4
// BE-NEXT: ret void
void test_aligned_bes32(__INT_LEAST32_TYPE__ v, unsigned char *p) { stdc_store8_aligned_bes32(v, p); }

// LE-LABEL: @test_aligned_bes64(
// LE:      %0 = tail call i64 @llvm.bswap.i64(i64 %v)
// LE-NEXT: store i64 %0, ptr %p, align 8
// LE-NEXT: ret void
// BE-LABEL: @test_aligned_bes64(
// BE:      store i64 %v, ptr %p, align 8
// BE-NEXT: ret void
void test_aligned_bes64(__INT_LEAST64_TYPE__ v, unsigned char *p) { stdc_store8_aligned_bes64(v, p); }

// A pointer with known alignment keeps it for the unaligned variants...
alignas(4) unsigned char buf4[4];
// LE-LABEL: @test_leu32_known_alignment(
// LE:      store i32 %v, ptr @buf4, align 4
// LE-NEXT: ret void
// BE-LABEL: @test_leu32_known_alignment(
// BE:      %0 = tail call i32 @llvm.bswap.i32(i32 %v)
// BE-NEXT: store i32 %0, ptr @buf4, align 4
// BE-NEXT: ret void
void test_leu32_known_alignment(__UINT_LEAST32_TYPE__ v) { stdc_store8_leu32(v, buf4); }

// ...and the aligned variants never lower it below the natural alignment.
alignas(16) unsigned char buf16[16];
// LE-LABEL: @test_aligned_leu64_known_alignment(
// LE:      store i64 %v, ptr @buf16, align 16
// LE-NEXT: ret void
// BE-LABEL: @test_aligned_leu64_known_alignment(
// BE:      %0 = tail call i64 @llvm.bswap.i64(i64 %v)
// BE-NEXT: store i64 %0, ptr @buf16, align 16
// BE-NEXT: ret void
void test_aligned_leu64_known_alignment(__UINT_LEAST64_TYPE__ v) { stdc_store8_aligned_leu64(v, buf16); }

// The result is void and the arguments are each evaluated exactly once.
__UINT_LEAST32_TYPE__ next_value(void);
unsigned char *next_ptr(void);
// LE-LABEL: @test_single_evaluation(
// LE:      %call = tail call i32 @next_value()
// LE-NEXT: %call1 = tail call ptr @next_ptr()
// LE-NEXT: %0 = tail call i32 @llvm.bswap.i32(i32 %call)
// LE-NEXT: store i32 %0, ptr %call1, align 1{{$}}
// LE-NEXT: ret void
// BE-LABEL: @test_single_evaluation(
// BE:      %call = tail call zeroext i32 @next_value()
// BE-NEXT: %call1 = tail call ptr @next_ptr()
// BE-NEXT: store i32 %call, ptr %call1, align 1{{$}}
// BE-NEXT: ret void
void test_single_evaluation(void) { stdc_store8_beu32(next_value(), next_ptr()); }

// The destination may be a variable-length array.
void use_buf(unsigned char *);
// LE-LABEL: @test_vla(
// LE:      %0 = zext i32 %n to i64
// LE-NEXT: %vla = alloca i8, i64 %0, align 16
// LE-NEXT: %1 = tail call i32 @llvm.bswap.i32(i32 %v)
// LE-NEXT: store i32 %1, ptr %vla, align 16
// LE-NEXT: call void @use_buf(ptr noundef nonnull %vla)
// LE-NEXT: ret void
// BE-LABEL: @test_vla(
// BE:      %0 = zext i32 %n to i64
// BE-NEXT: %vla = alloca i8, i64 %0, align 4
// BE-NEXT: store i32 %v, ptr %vla, align 4
// BE-NEXT: call void @use_buf(ptr noundef nonnull %vla)
// BE-NEXT: ret void
void test_vla(int n, __UINT_LEAST32_TYPE__ v) {
  unsigned char buf[n];
  stdc_store8_beu32(v, buf);
  use_buf(buf);
}

// The __builtin_ spellings lower exactly like the stdc_ ones.
// LE-LABEL: @test_builtin_leu16(
// LE:      store i16 %v, ptr %p, align 1{{$}}
// LE-NEXT: ret void
// BE-LABEL: @test_builtin_leu16(
// BE:      %0 = tail call i16 @llvm.bswap.i16(i16 %v)
// BE-NEXT: store i16 %0, ptr %p, align 1{{$}}
// BE-NEXT: ret void
void test_builtin_leu16(__UINT_LEAST16_TYPE__ v, unsigned char *p) { __builtin_stdc_store8_leu16(v, p); }

// LE-LABEL: @test_builtin_beu32(
// LE:      %0 = tail call i32 @llvm.bswap.i32(i32 %v)
// LE-NEXT: store i32 %0, ptr %p, align 1{{$}}
// LE-NEXT: ret void
// BE-LABEL: @test_builtin_beu32(
// BE:      store i32 %v, ptr %p, align 1{{$}}
// BE-NEXT: ret void
void test_builtin_beu32(__UINT_LEAST32_TYPE__ v, unsigned char *p) { __builtin_stdc_store8_beu32(v, p); }

// LE-LABEL: @test_builtin_aligned_bes64(
// LE:      %0 = tail call i64 @llvm.bswap.i64(i64 %v)
// LE-NEXT: store i64 %0, ptr %p, align 8
// LE-NEXT: ret void
// BE-LABEL: @test_builtin_aligned_bes64(
// BE:      store i64 %v, ptr %p, align 8
// BE-NEXT: ret void
void test_builtin_aligned_bes64(__INT_LEAST64_TYPE__ v, unsigned char *p) { __builtin_stdc_store8_aligned_bes64(v, p); }

// LE-LABEL: @test_builtin_les8(
// LE:      store i8 %v, ptr %p, align 1{{$}}
// LE-NEXT: ret void
// BE-LABEL: @test_builtin_les8(
// BE:      store i8 %v, ptr %p, align 1{{$}}
// BE-NEXT: ret void
void test_builtin_les8(__INT_LEAST8_TYPE__ v, unsigned char *p) { __builtin_stdc_store8_les8(v, p); }

// The alignment of the destination is derived from the pointer expression.
alignas(8) unsigned char buf8[16];
struct S { char c; alignas(4) unsigned char b[8]; };
struct S gs;

// LE-LABEL: @test_array_offset_2(
// LE:      store i16 %v, ptr getelementptr inbounds nuw (i8, ptr @buf8, i64 2), align 2
// LE-NEXT: ret void
// BE-LABEL: @test_array_offset_2(
// BE:      %0 = tail call i16 @llvm.bswap.i16(i16 %v)
// BE-NEXT: store i16 %0, ptr getelementptr inbounds nuw (i8, ptr @buf8, i64 2), align 2
// BE-NEXT: ret void
void test_array_offset_2(__UINT_LEAST16_TYPE__ v) { stdc_store8_leu16(v, buf8 + 2); }

// LE-LABEL: @test_array_offset_3(
// LE:      store i16 %v, ptr getelementptr inbounds nuw (i8, ptr @buf8, i64 3), align 1{{$}}
// LE-NEXT: ret void
// BE-LABEL: @test_array_offset_3(
// BE:      %0 = tail call i16 @llvm.bswap.i16(i16 %v)
// BE-NEXT: store i16 %0, ptr getelementptr inbounds nuw (i8, ptr @buf8, i64 3), align 1{{$}}
// BE-NEXT: ret void
void test_array_offset_3(__UINT_LEAST16_TYPE__ v) { stdc_store8_leu16(v, buf8 + 3); }

// LE-LABEL: @test_member(
// LE:      store i32 %v, ptr getelementptr inbounds nuw (i8, ptr @gs, i64 4), align 4
// LE-NEXT: ret void
// BE-LABEL: @test_member(
// BE:      %0 = tail call i32 @llvm.bswap.i32(i32 %v)
// BE-NEXT: store i32 %0, ptr getelementptr inbounds nuw (i8, ptr @gs, i64 4), align 4
// BE-NEXT: ret void
void test_member(__UINT_LEAST32_TYPE__ v) { stdc_store8_leu32(v, gs.b); }

// LE-LABEL: @test_member_offset_4(
// LE:      store i32 %v, ptr getelementptr inbounds nuw (i8, ptr @gs, i64 8), align 4
// LE-NEXT: ret void
// BE-LABEL: @test_member_offset_4(
// BE:      %0 = tail call i32 @llvm.bswap.i32(i32 %v)
// BE-NEXT: store i32 %0, ptr getelementptr inbounds nuw (i8, ptr @gs, i64 8), align 4
// BE-NEXT: ret void
void test_member_offset_4(__UINT_LEAST32_TYPE__ v) { stdc_store8_leu32(v, gs.b + 4); }

// With constant arguments the byte swap folds into the stored constant.
// LE-LABEL: @test_const_leu32(
// LE:      store i32 305419896, ptr %p, align 1{{$}}
// LE-NEXT: ret void
// BE-LABEL: @test_const_leu32(
// BE:      store i32 2018915346, ptr %p, align 1{{$}}
// BE-NEXT: ret void
void test_const_leu32(unsigned char *p) { stdc_store8_leu32(0x12345678u, p); }

// LE-LABEL: @test_const_beu32(
// LE:      store i32 2018915346, ptr %p, align 1{{$}}
// LE-NEXT: ret void
// BE-LABEL: @test_const_beu32(
// BE:      store i32 305419896, ptr %p, align 1{{$}}
// BE-NEXT: ret void
void test_const_beu32(unsigned char *p) { stdc_store8_beu32(0x12345678u, p); }

// LE-LABEL: @test_const_bes16(
// LE:      store i16 -257, ptr %p, align 1{{$}}
// LE-NEXT: ret void
// BE-LABEL: @test_const_bes16(
// BE:      store i16 -2, ptr %p, align 1{{$}}
// BE-NEXT: ret void
void test_const_bes16(unsigned char *p) { stdc_store8_bes16(-2, p); }

// LE-LABEL: @test_const_les16(
// LE:      store i16 -2, ptr %p, align 1{{$}}
// LE-NEXT: ret void
// BE-LABEL: @test_const_les16(
// BE:      store i16 -257, ptr %p, align 1{{$}}
// BE-NEXT: ret void
void test_const_les16(unsigned char *p) { stdc_store8_les16(-2, p); }

// A big-endian store followed by a big-endian load folds to the identity.
// LE-LABEL: @test_roundtrip(
// LE:      ret i32 %v
// BE-LABEL: @test_roundtrip(
// BE:      ret i32 %v
__UINT_LEAST32_TYPE__ test_roundtrip(__UINT_LEAST32_TYPE__ v) {
  unsigned char b[4];
  stdc_store8_beu32(v, b);
  return stdc_load8_beu32(b);
}

// The store is visible to a load of the same bytes through another type.
// LE-LABEL: @test_visible_through_int(
// LE:      store i32 16909060, ptr %ip, align 1{{$}}
// LE-NEXT: ret i32 16909060
// BE-LABEL: @test_visible_through_int(
// BE:      store i32 67305985, ptr %ip, align 1{{$}}
// BE-NEXT: ret i32 67305985
unsigned test_visible_through_int(unsigned *ip) {
  stdc_store8_leu32(0x01020304u, (unsigned char *)ip);
  return *ip;
}
