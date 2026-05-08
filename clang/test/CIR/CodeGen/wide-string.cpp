// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

// Test with built-in char16_t type
const char16_t *test_utf16() {
  return u"你好世界";
}

// CIR: cir.global "private" constant cir_private dso_local @{{.+}} = #cir.const_array<[#cir.int<20320> : !u16i, #cir.int<22909> : !u16i, #cir.int<19990> : !u16i, #cir.int<30028> : !u16i, #cir.int<0> : !u16i]> : !cir.array<!u16i x 5>
// LLVM: @{{.+}} = private constant [5 x i16] [i16 20320, i16 22909, i16 19990, i16 30028, i16 0]
// OGCG: @{{.+}} = private unnamed_addr constant [5 x i16] [i16 20320, i16 22909, i16 19990, i16 30028, i16 0]

const char32_t *test_utf32() {
  return U"你好世界";
}

// CIR: cir.global "private" constant cir_private dso_local @{{.+}} = #cir.const_array<[#cir.int<20320> : !u32i, #cir.int<22909> : !u32i, #cir.int<19990> : !u32i, #cir.int<30028> : !u32i, #cir.int<0> : !u32i]> : !cir.array<!u32i x 5>
// LLVM: @{{.+}} = private constant [5 x i32] [i32 20320, i32 22909, i32 19990, i32 30028, i32 0]
// OGCG: @{{.+}} = private unnamed_addr constant [5 x i32] [i32 20320, i32 22909, i32 19990, i32 30028, i32 0]

const char16_t *test_zero16() {
  return u"\0\0\0\0";
}

// CIR: cir.global "private" constant cir_private dso_local @{{.+}} = #cir.zero : !cir.array<!u16i x 5>
// LLVM: @{{.+}} = private constant [5 x i16] zeroinitializer
// OGCG: @{{.+}} = private unnamed_addr constant [5 x i16] zeroinitializer

const char32_t *test_zero32() {
  return U"\0\0\0\0";
}

// CIR: cir.global "private" constant cir_private dso_local @{{.+}} = #cir.zero : !cir.array<!u32i x 5>
// LLVM: @{{.+}} = private constant [5 x i32] zeroinitializer
// OGCG: @{{.+}} = private unnamed_addr constant [5 x i32] zeroinitializer

const wchar_t *test_wchar() {
  return L"1234";
}

// CIR: cir.global "private" constant cir_private dso_local @{{.+}} = #cir.const_array<[#cir.int<49> : !s32i, #cir.int<50> : !s32i, #cir.int<51> : !s32i, #cir.int<52> : !s32i, #cir.int<0> : !s32i]> : !cir.array<!s32i x 5>
// LLVM: @{{.+}} = private constant [5 x i32] [i32 49, i32 50, i32 51, i32 52, i32 0]
// OGCG: @{{.+}} = private unnamed_addr constant [5 x i32] [i32 49, i32 50, i32 51, i32 52, i32 0]

const wchar_t *test_wchar_zero() {
  return L"";
}

// CIR: cir.global "private" constant cir_private dso_local @{{.+}} = #cir.zero : !cir.array<!s32i x 1>
// LLVM: @{{.+}} = private constant [1 x i32] zeroinitializer
// OGCG: @{{.+}} = private unnamed_addr constant [1 x i32] zeroinitializer

const char16_t *test_char16_typedef() {
  return u"test";
}

// CIR: cir.global "private" constant cir_private dso_local @{{.+}} = #cir.const_array<[#cir.int<116> : !u16i, #cir.int<101> : !u16i, #cir.int<115> : !u16i, #cir.int<116> : !u16i, #cir.int<0> : !u16i]> : !cir.array<!u16i x 5>
// LLVM: @{{.+}} = private constant [5 x i16] [i16 116, i16 101, i16 115, i16 116, i16 0]
// OGCG: @{{.+}} = private unnamed_addr constant [5 x i16] [i16 116, i16 101, i16 115, i16 116, i16 0]
