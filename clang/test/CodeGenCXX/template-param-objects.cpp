// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++20 %s -emit-llvm -o - | FileCheck %s --check-prefixes=ITANIUM,CHECK
// RUN: %clang_cc1 -triple x86_64-windows -std=c++20 %s -emit-llvm -o - | FileCheck %s --check-prefixes=MSABI,CHECK

struct S { char buf[32]; };
template<S s> constexpr const char *begin() { return s.buf; }
template<S s> constexpr const char *end() { return s.buf + __builtin_strlen(s.buf); }

// ITANIUM: [[HELLO:@_ZTAXtl1StlA32_cLc104ELc101ELc108ELc108ELc111ELc32ELc119ELc111ELc114ELc108ELc100EEEE]]
// MSABI: [[HELLO:@"[?][?]__N2US@@3D0GI@@0GF@@0GM@@0GM@@0GP@@0CA@@0HH@@0GP@@0HC@@0GM@@0GE@@0A@@0A@@0A@@0A@@0A@@0A@@0A@@0A@@0A@@0A@@0A@@0A@@0A@@0A@@0A@@0A@@0A@@0A@@0A@@0A@@0A@@@@@"]]
// CHECK-SAME: = linkonce_odr constant { <{ [11 x i8], [21 x i8] }> } { <{ [11 x i8], [21 x i8] }> <{ [11 x i8] c"hello world", [21 x i8] zeroinitializer }> }, comdat

// ITANIUM: @p
// MSABI: @"?p@@3PEBDEB"
// CHECK-SAME: global i8* getelementptr inbounds ({{.*}}* [[HELLO]], i32 0, i32 0, i32 0, i32 0)
const char *p = begin<S{"hello world"}>();
// ITANIUM: @q
// MSABI: @"?q@@3PEBDEB"
// CHECK-SAME: global i8* getelementptr ({{.*}}* [[HELLO]], i32 0, i32 0, i32 0, i64 11)
const char *q = end<S{"hello world"}>();
