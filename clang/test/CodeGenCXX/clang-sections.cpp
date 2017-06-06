// RUN: %clang_cc1 -emit-llvm -triple arm-none-eabi -o - %s | FileCheck %s
// Test that global variables, statics and functions are attached section-attributes
// as per '#pragma clang section' directives.

extern "C" {
// test with names for each section
#pragma clang section bss="my_bss.1" data="my_data.1" rodata="my_rodata.1"
#pragma clang section text="my_text.1"
int a;      // my_bss.1
int b = 1;  // my_data.1
int c[4];   // my_bss.1
short d[5] = {0}; // my_bss.1
short e[6] = {0, 0, 1}; // my_data.1
extern const int f;
const int f = 2;  // my_rodata.1
int foo(void) {   // my_text.1
  return b;
}
static int g[2]; // my_bss.1
#pragma clang section bss=""
int h; // default - .bss
#pragma clang section data=""  bss="my_bss.2" text="my_text.2"
int i = 0; // my_bss.2
extern const int j;
const int j = 4; // default - .rodata
int k; // my_bss.2
extern int zoo(int *x, int *y);
int goo(void) {  // my_text.2
  static int lstat_h;  // my_bss.2
  return zoo(g, &lstat_h);
}
#pragma clang section rodata="my_rodata.2" data="my_data.2"
int l = 5; // my_data.2
extern const int m;
const int m = 6; // my_rodata.2
#pragma clang section rodata="" data="" bss="" text=""
int n; // default
int o = 6; // default
extern const int p;
const int p = 7; // default
int hoo(void) {
  return b;
}
}
//CHECK: @a = global i32 0, align 4 #0
//CHECK: @b = global i32 1, align 4 #0
//CHECK: @c = global [4 x i32] zeroinitializer, align 4 #0
//CHECK: @d = global [5 x i16] zeroinitializer, align 2 #0
//CHECK: @e = global [6 x i16] [i16 0, i16 0, i16 1, i16 0, i16 0, i16 0], align 2 #0
//CHECK: @f = constant i32 2, align 4 #0

//CHECK: @h = global i32 0, align 4 #1
//CHECK: @i = global i32 0, align 4 #2
//CHECK: @j = constant i32 4, align 4 #2
//CHECK: @k = global i32 0, align 4 #2
//CHECK: @_ZZ3gooE7lstat_h = internal global i32 0, align 4 #2
//CHECK: @_ZL1g = internal global [2 x i32] zeroinitializer, align 4 #0

//CHECK: @l = global i32 5, align 4 #3
//CHECK: @m = constant i32 6, align 4 #3

//CHECK: @n = global i32 0, align 4
//CHECK: @o = global i32 6, align 4
//CHECK: @p = constant i32 7, align 4

//CHECK: define i32 @foo() #4 {
//CHECK: define i32 @goo() #5 {
//CHECK: declare i32 @zoo(i32*, i32*) #6
//CHECK: define i32 @hoo() #7 {

//CHECK: attributes #0 = { "bss-section"="my_bss.1" "data-section"="my_data.1" "rodata-section"="my_rodata.1" }
//CHECK: attributes #1 = { "data-section"="my_data.1" "rodata-section"="my_rodata.1" }
//CHECK: attributes #2 = { "bss-section"="my_bss.2" "rodata-section"="my_rodata.1" }
//CHECK: attributes #3 = { "bss-section"="my_bss.2" "data-section"="my_data.2" "rodata-section"="my_rodata.2" }
//CHECK: attributes #4 = { {{.*"implicit-section-name"="my_text.1".*}} }
//CHECK: attributes #5 = { {{.*"implicit-section-name"="my_text.2".*}} }
//CHECK-NOT: attributes #6 = { {{.*"implicit-section-name".*}} }
//CHECK-NOT: attributes #7 = { {{.*"implicit-section-name".*}} }
