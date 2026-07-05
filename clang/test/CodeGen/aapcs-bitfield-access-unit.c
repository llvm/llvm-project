// RUN: %clang_cc1 -triple armv8-none-linux-eabi   -fno-aapcs-bitfield-width -fdump-record-layouts-simple -emit-llvm -o /dev/null %s | FileCheck %s -check-prefixes=LAYOUT
// RUN: %clang_cc1 -triple armebv8-none-linux-eabi   -fno-aapcs-bitfield-width -fdump-record-layouts-simple -emit-llvm -o /dev/null %s | FileCheck %s -check-prefixes=LAYOUT

// RUN: %clang_cc1 -triple armv8-none-linux-eabi   -faapcs-bitfield-width -fdump-record-layouts-simple -emit-llvm -o /dev/null %s | FileCheck %s -check-prefixes=LAYOUT
// RUN: %clang_cc1 -triple armebv8-none-linux-eabi   -faapcs-bitfield-width -fdump-record-layouts-simple -emit-llvm -o /dev/null %s | FileCheck %s -check-prefixes=LAYOUT

struct st0 {
  short c : 7;
} st0;
// LAYOUT-LABEL: LLVMType:%struct.st0 =
// LAYOUT-SAME: type { i8, i8 }
// LAYOUT: BitFields:[
// LAYOUT-NEXT: <CGBitFieldInfo Offset:{{[0-9]+}} Size:7 IsSigned:1 StorageSize:8 StorageOffset:0
// LAYOUT-NEXT: ]>

struct st1 {
  int a : 10;
  short c : 6;
} st1;
// LAYOUT-LABEL: LLVMType:%struct.st1 =
// LAYOUT-SAME: type { i16, [2 x i8] }
// LAYOUT: BitFields:[
// LAYOUT-NEXT: <CGBitFieldInfo Offset:{{[0-9]+}} Size:10 IsSigned:1 StorageSize:16 StorageOffset:0
// LAYOUT-NEXT: <CGBitFieldInfo Offset:{{[0-9]+}} Size:6 IsSigned:1 StorageSize:16 StorageOffset:0
// LAYOUT-NEXT: ]>

struct st2 {
  int a : 10;
  short c : 7;
} st2;
// LAYOUT-LABEL: LLVMType:%struct.st2 =
// LAYOUT-SAME: type { i32 }
// LAYOUT: BitFields:[
// LAYOUT-NEXT: <CGBitFieldInfo Offset:{{[0-9]+}} Size:10 IsSigned:1 StorageSize:32 StorageOffset:0
// LAYOUT-NEXT: <CGBitFieldInfo Offset:{{[0-9]+}} Size:7 IsSigned:1 StorageSize:32 StorageOffset:0
// LAYOUT-NEXT: ]>

struct st3 {
  volatile short c : 7;
} st3;
// LAYOUT-LABEL: LLVMType:%struct.st3 =
// LAYOUT-SAME: type { i8, i8 }
// LAYOUT: BitFields:[
// LAYOUT-NEXT: <CGBitFieldInfo Offset:{{[0-9]+}} Size:7 IsSigned:1 StorageSize:8 StorageOffset:0
// LAYOUT-NEXT: ]>

struct st4 {
  int b : 9;
  volatile char c : 5;
} st4;
// LAYOUT-LABEL: LLVMType:%struct.st4 =
// LAYOUT-SAME: type { i16, [2 x i8] }
// LAYOUT: BitFields:[
// LAYOUT-NEXT: <CGBitFieldInfo Offset:{{[0-9]+}} Size:9 IsSigned:1 StorageSize:16 StorageOffset:0
// LAYOUT-NEXT: <CGBitFieldInfo Offset:{{[0-9]+}} Size:5 IsSigned:1 StorageSize:16 StorageOffset:0
// LAYOUT-NEXT: ]>

struct st5 {
  int a : 12;
  volatile char c : 5;
} st5;
// LAYOUT-LABEL: LLVMType:%struct.st5 =
// LAYOUT-SAME: type { i32 }
// LAYOUT: BitFields:[
// LAYOUT-NEXT: <CGBitFieldInfo Offset:{{[0-9]+}} Size:12 IsSigned:1 StorageSize:32 StorageOffset:0
// LAYOUT-NEXT: <CGBitFieldInfo Offset:{{[0-9]+}} Size:5 IsSigned:1 StorageSize:32 StorageOffset:0
// LAYOUT-NEXT: ]>

struct st6 {
  int a : 12;
  char b;
  int c : 5;
} st6;
// LAYOUT-LABEL: LLVMType:%struct.st6 =
// LAYOUT-SAME: type { i16, i8, i8 }
// LAYOUT: BitFields:[
// LAYOUT-NEXT: <CGBitFieldInfo Offset:{{[0-9]+}} Size:12 IsSigned:1 StorageSize:16 StorageOffset:0
// LAYOUT-NEXT: <CGBitFieldInfo Offset:{{[0-9]+}} Size:5 IsSigned:1 StorageSize:8 StorageOffset:3
// LAYOUT-NEXT: ]>

struct st7a {
  char a;
  int b : 5;
} st7a;
// LAYOUT-LABEL: LLVMType:%struct.st7a =
// LAYOUT-SAME: type { i8, i8, [2 x i8] }
// LAYOUT: BitFields:[
// LAYOUT-NEXT: <CGBitFieldInfo Offset:{{[0-9]+}} Size:5 IsSigned:1 StorageSize:8 StorageOffset:1
// LAYOUT-NEXT: ]>

struct st7b {
  char x;
  volatile struct st7a y;
} st7b;
// LAYOUT-LABEL: LLVMType:%struct.st7b =
// LAYOUT-SAME: type { i8, [3 x i8], %struct.st7a }
// LAYOUT: BitFields:[
// LAYOUT-NEXT: ]>

struct st8 {
  unsigned f : 16;
} st8;
// LAYOUT-LABEL: LLVMType:%struct.st8 =
// LAYOUT-SAME: type { i16, [2 x i8] }
// LAYOUT: BitFields:[
// LAYOUT-NEXT: <CGBitFieldInfo Offset:{{[0-9]+}} Size:16 IsSigned:0 StorageSize:16 StorageOffset:0
// LAYOUT-NEXT: ]>

struct st9{
  int f : 8;
} st9;
// LAYOUT-LABEL: LLVMType:%struct.st9 =
// LAYOUT-SAME: type { i8, [3 x i8] }
// LAYOUT: BitFields:[
// LAYOUT-NEXT: <CGBitFieldInfo Offset:{{[0-9]+}} Size:8 IsSigned:1 StorageSize:8 StorageOffset:0
// LAYOUT-NEXT: ]>

struct st10{
  int e : 1;
  int f : 8;
} st10;
// LAYOUT-LABEL: LLVMType:%struct.st10 =
// LAYOUT-SAME: type { i16, [2 x i8] }
// LAYOUT: BitFields:[
// LAYOUT-NEXT: <CGBitFieldInfo Offset:{{[0-9]+}} Size:1 IsSigned:1 StorageSize:16 StorageOffset:0
// LAYOUT-NEXT: <CGBitFieldInfo Offset:{{[0-9]+}} Size:8 IsSigned:1 StorageSize:16 StorageOffset:0
// LAYOUT-NEXT: ]>

struct st11{
  char e;
  int f : 16;
} st11;
// LAYOUT-LABEL: LLVMType:%struct.st11 =
// LAYOUT-SAME: type <{ i8, i16, i8 }>
// LAYOUT: BitFields:[
// LAYOUT-NEXT: <CGBitFieldInfo Offset:{{[0-9]+}} Size:16 IsSigned:1 StorageSize:16 StorageOffset:1
// LAYOUT-NEXT: ]>

struct st12{
  int e : 8;
  int f : 16;
} st12;
// LAYOUT-LABEL: LLVMType:%struct.st12 =
// LAYOUT-SAME: type { i32 }
// LAYOUT: BitFields:[
// LAYOUT-NEXT: <CGBitFieldInfo Offset:{{[0-9]+}} Size:8 IsSigned:1 StorageSize:32 StorageOffset:0
// LAYOUT-NEXT: <CGBitFieldInfo Offset:{{[0-9]+}} Size:16 IsSigned:1 StorageSize:32 StorageOffset:0
// LAYOUT-NEXT: ]>

struct st13 {
  char a : 8;
  int b : 32;
} __attribute__((packed)) st13;
// LAYOUT-LABEL: LLVMType:%struct.st13 =
// LAYOUT-SAME: type <{ i8, i32 }>
// LAYOUT: BitFields:[
// LAYOUT-NEXT: <CGBitFieldInfo Offset:{{[0-9]+}} Size:8 IsSigned:1 StorageSize:8 StorageOffset:0
// LAYOUT-NEXT: <CGBitFieldInfo Offset:{{[0-9]+}} Size:32 IsSigned:1 StorageSize:32 StorageOffset:1
// LAYOUT-NEXT: ]>

struct st14 {
  char a : 8;
} __attribute__((packed)) st14;
// LAYOUT-LABEL: LLVMType:%struct.st14 =
// LAYOUT-SAME: type { i8 }
// LAYOUT: BitFields:[
// LAYOUT-NEXT: <CGBitFieldInfo Offset:{{[0-9]+}} Size:8 IsSigned:1 StorageSize:8 StorageOffset:0
// LAYOUT-NEXT: ]>

struct st15 {
  short a : 8;
} __attribute__((packed)) st15;
// LAYOUT-LABEL: LLVMType:%struct.st15 =
// LAYOUT-SAME: type { i8 }
// LAYOUT: BitFields:[
// LAYOUT-NEXT: <CGBitFieldInfo Offset:{{[0-9]+}} Size:8 IsSigned:1 StorageSize:8 StorageOffset:0
// LAYOUT-NEXT: ]>

struct st16 {
  int a : 32;
  int b : 16;
  int c : 32;
  int d : 16;
} st16;
// LAYOUT-LABEL: LLVMType:%struct.st16 =
// LAYOUT-SAME: type { i32, i16, i32, i16 }
// LAYOUT: BitFields:[
// LAYOUT-NEXT: <CGBitFieldInfo Offset:{{[0-9]+}} Size:32 IsSigned:1 StorageSize:32 StorageOffset:0
// LAYOUT-NEXT: <CGBitFieldInfo Offset:{{[0-9]+}} Size:16 IsSigned:1 StorageSize:16 StorageOffset:4
// LAYOUT-NEXT: <CGBitFieldInfo Offset:{{[0-9]+}} Size:32 IsSigned:1 StorageSize:32 StorageOffset:8
// LAYOUT-NEXT: <CGBitFieldInfo Offset:{{[0-9]+}} Size:16 IsSigned:1 StorageSize:16 StorageOffset:12
// LAYOUT-NEXT: ]>

struct st17 {
int b : 32;
char c : 8;
} __attribute__((packed)) st17;
// LAYOUT-LABEL: LLVMType:%struct.st17 =
// LAYOUT-SAME: type <{ i32, i8 }>
// LAYOUT: BitFields:[
// LAYOUT-NEXT: <CGBitFieldInfo Offset:{{[0-9]+}} Size:32 IsSigned:1 StorageSize:32 StorageOffset:0
// LAYOUT-NEXT: <CGBitFieldInfo Offset:{{[0-9]+}} Size:8 IsSigned:1 StorageSize:8 StorageOffset:4
// LAYOUT-NEXT: ]>

struct zero_bitfield {
  int a : 8;
  char : 0;
  int b : 8;
} st18;
// LAYOUT-LABEL: LLVMType:%struct.zero_bitfield =
// LAYOUT-SAME: type { i8, i8, [2 x i8] }
// LAYOUT: BitFields:[
// LAYOUT-NEXT: <CGBitFieldInfo Offset:{{[0-9]+}} Size:8 IsSigned:1 StorageSize:8 StorageOffset:0
// LAYOUT-NEXT: <CGBitFieldInfo Offset:{{[0-9]+}} Size:8 IsSigned:1 StorageSize:8 StorageOffset:1
// LAYOUT-NEXT: ]>

struct zero_bitfield_ok {
  short a : 8;
  char a1 : 8;
  long : 0;
  int b : 24;
} st19;
// LAYOUT-LABEL: LLVMType:%struct.zero_bitfield_ok =
// LAYOUT-SAME: type { i16, i32 }
// LAYOUT: BitFields:[
// LAYOUT-NEXT: <CGBitFieldInfo Offset:{{[0-9]+}} Size:8 IsSigned:1 StorageSize:16 StorageOffset:0
// LAYOUT-NEXT: <CGBitFieldInfo Offset:{{[0-9]+}} Size:8 IsSigned:1 StorageSize:16 StorageOffset:0
// LAYOUT-NEXT: <CGBitFieldInfo Offset:{{[0-9]+}} Size:24 IsSigned:1 StorageSize:32 StorageOffset:4
// LAYOUT-NEXT: ]>


