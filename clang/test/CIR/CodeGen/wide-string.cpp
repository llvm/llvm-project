// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

const char16_t *test_utf16() {
  return u"你好世界";
}

// CHECK: cir.global "private" constant internal @{{.+}} = #cir.const_array<[#cir.int<20320> : !u16i, #cir.int<22909> : !u16i, #cir.int<19990> : !u16i, #cir.int<30028> : !u16i, #cir.int<0> : !u16i]> : !cir.array<!u16i x 5>

const char32_t *test_utf32() {
  return U"你好世界";
}

// CHECK: cir.global "private" constant internal @{{.+}} = #cir.const_array<[#cir.int<20320> : !u32i, #cir.int<22909> : !u32i, #cir.int<19990> : !u32i, #cir.int<30028> : !u32i, #cir.int<0> : !u32i]> : !cir.array<!u32i x 5>

const char16_t *test_zero16() {
  return u"\0\0\0\0";
}

// CHECK: cir.global "private" constant internal @{{.+}} = #cir.zero : !cir.array<!u16i x 5>

const char32_t *test_zero32() {
  return U"\0\0\0\0";
}

// CHECK: cir.global "private" constant internal @{{.+}} = #cir.zero : !cir.array<!u32i x 5>
