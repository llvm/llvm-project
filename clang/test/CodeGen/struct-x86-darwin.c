// RUN: %clang_cc1 %s -emit-llvm -triple=i686-apple-darwin9 -o - | FileCheck %s
// CHECK: STest1 = type { i32, [4 x i16], double }
// CHECK: STest2 = type { i16, i16, i32, i32 }
// CHECK: STest3 = type { i8, i16, i32 }
// CHECK: STestB1 = type { i8, i8 }
// CHECK: STestB2 = type { i8, i8, i8 }
// CHECK: STestB3 = type { i8, i8 }
// CHECK: STestB4 = type { i8, i8, i8, i8 }
// CHECK: STestB5 = type { i8, i16, i8 }
// CHECK: STestB6 = type { i8, i8, i16 }
// Test struct layout for x86-darwin target

struct STest1 {int x; short y[4]; double z; } st1;
struct STest2 {short a,b; int c,d; } st2;
struct STest3 {char a; short b; int c; } st3;

// Bitfields 
struct STestB1 {char a; char b:2; } stb1;
struct STestB2 {char a; char b:5; char c:4; } stb2;
struct STestB3 {char a; char b:2; } stb3;
struct STestB4 {char a; short b:2; char c; } stb4;
struct STestB5 {char a; short b:10; char c; } stb5;
struct STestB6 {int a:1; char b; int c:13 } stb6;

// Packed struct STestP1 {char a; short b; int c; } __attribute__((__packed__)) stp1;
