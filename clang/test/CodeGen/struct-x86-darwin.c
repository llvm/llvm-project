// RUN: %clang_cc1 %s -emit-llvm -o /dev/null -triple=i686-apple-darwin9 -fdump-record-layouts-simple | FileCheck %s

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
struct STestB6 {int a:1; char b; int c:13; } stb6;

// Packed struct STestP1 {char a; short b; int c; } __attribute__((__packed__)) stp1;

// CHECK-LABEL: LLVMType:%struct.STest1 =
// CHECK-SAME: type { i32, [4 x i16], double }
// CHECK: BitFields:[
// CHECK-NEXT: ]>

// CHECK-LABEL: LLVMType:%struct.STest2 =
// CHECK-SAME: type { i16, i16, i32, i32 }
// CHECK: BitFields:[
// CHECK-NEXT: ]>

// CHECK-LABEL: LLVMType:%struct.STest3 =
// CHECK-SAME: type { i8, i16, i32 }
// CHECK: BitFields:[
// CHECK-NEXT: ]>

// CHECK-LABEL: LLVMType:%struct.STestB1 =
// CHECK-SAME: type { i8, i8 }
// CHECK: BitFields:[
// CHECK-NEXT: <CGBitFieldInfo Offset:0 Size:2 IsSigned:1 StorageSize:8 StorageOffset:1
// CHECK-NEXT: ]>

// CHECK-LABEL: LLVMType:%struct.STestB2 =
// CHECK-SAME: type <{ i8, i16 }>
// CHECK: BitFields:[
// CHECK-NEXT: <CGBitFieldInfo Offset:0 Size:5 IsSigned:1 StorageSize:16 StorageOffset:1
// CHECK-NEXT: <CGBitFieldInfo Offset:8 Size:4 IsSigned:1 StorageSize:16 StorageOffset:1
// CHECK-NEXT: ]>

// CHECK-LABEL: LLVMType:%struct.STestB3 =
// CHECK-SAME: type { i8, i8 }
// CHECK: BitFields:[
// CHECK-NEXT: <CGBitFieldInfo Offset:0 Size:2 IsSigned:1 StorageSize:8 StorageOffset:1
// CHECK-NEXT: ]>

// CHECK-LABEL: LLVMType:%struct.STestB4 =
// CHECK-SAME: type { i8, i8, i8, i8 }
// CHECK: BitFields:[
// CHECK-NEXT: <CGBitFieldInfo Offset:0 Size:2 IsSigned:1 StorageSize:8 StorageOffset:1
// CHECK-NEXT: ]>

// CHECK-LABEL: LLVMType:%struct.STestB5 =
// CHECK-SAME: type { i8, i16, i8 }
// CHECK: BitFields:[
// CHECK-NEXT: <CGBitFieldInfo Offset:0 Size:10 IsSigned:1 StorageSize:16 StorageOffset:2
// CHECK-NEXT: ]>

// CHECK-LABEL: LLVMType:%struct.STestB6 =
// CHECK-SAME: type { i8, i8, i16 }
// CHECK: BitFields:[
// CHECK-NEXT: <CGBitFieldInfo Offset:0 Size:1 IsSigned:1 StorageSize:8 StorageOffset:0
// CHECK-NEXT: <CGBitFieldInfo Offset:0 Size:13 IsSigned:1 StorageSize:16 StorageOffset:2
// CHECK-NEXT: ]>
