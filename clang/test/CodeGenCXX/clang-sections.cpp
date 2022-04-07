// RUN: %clang_cc1 -no-opaque-pointers -emit-llvm -triple arm-none-eabi -o - %s | FileCheck %s --check-prefixes=CHECK,ELF
// RUN: %clang_cc1 -no-opaque-pointers -emit-llvm -triple arm64-apple-ios -o - %s | FileCheck %s --check-prefixes=CHECK,MACHO
// Test that global variables, statics and functions are attached section-attributes
// as per '#pragma clang section' directives.

extern "C" {
// test with names for each section
#ifdef __MACH__
#pragma clang section bss = "__BSS,__mybss1" data = "__DATA,__mydata1" rodata = "__RODATA,__myrodata1"
#pragma clang section text = "__TEXT,__mytext1"
#else
#pragma clang section bss="my_bss.1" data="my_data.1" rodata="my_rodata.1"
#pragma clang section text="my_text.1"
#endif
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
#ifdef __MACH__
#pragma clang section data = "" bss = "__BSS,__mybss2" text = "__TEXT,__mytext2"
#else
#pragma clang section data=""  bss="my_bss.2" text="my_text.2"
#endif
int i = 0; // my_bss.2
extern const int j;
const int j = 4; // default - .rodata
int k; // my_bss.2
extern int zoo(int *x, int *y);
int goo(void) {  // my_text.2
  static int lstat_h;  // my_bss.2
  return zoo(g, &lstat_h);
}
#ifdef __MACH__
#pragma clang section rodata = "__RODATA,__myrodata2" data = "__DATA,__mydata2" relro = "__RELRO,__myrelro2"
#else
#pragma clang section rodata="my_rodata.2" data="my_data.2" relro="my_relro.2"
#endif
int l = 5; // my_data.2
extern const int m;
const int m = 6; // my_rodata.2

typedef int (*fptr_t)(void);
const fptr_t fptrs[2] = {&foo, &goo};
#pragma clang section rodata="" data="" bss="" text=""
int n; // default
int o = 6; // default
extern const int p;
const int p = 7; // default
int hoo(void) {
  return b + fptrs[f]();
}
}
//CHECK: @a ={{.*}} global i32 0, align 4 #0
//CHECK: @b ={{.*}} global i32 1, align 4 #0
//CHECK: @c ={{.*}} global [4 x i32] zeroinitializer, align 4 #0
//CHECK: @d ={{.*}} global [5 x i16] zeroinitializer, align 2 #0
//CHECK: @e ={{.*}} global [6 x i16] [i16 0, i16 0, i16 1, i16 0, i16 0, i16 0], align 2 #0
//CHECK: @f ={{.*}} constant i32 2, align 4 #0

//CHECK: @h ={{.*}} global i32 0, align 4 #1
//CHECK: @i ={{.*}} global i32 0, align 4 #2
//CHECK: @j ={{.*}} constant i32 4, align 4 #2
//CHECK: @k ={{.*}} global i32 0, align 4 #2
//CHECK: @_ZZ3gooE7lstat_h = internal global i32 0, align 4 #2
//CHECK: @_ZL1g = internal global [2 x i32] zeroinitializer, align 4 #0

//CHECK: @l ={{.*}} global i32 5, align 4 #3
//CHECK: @m ={{.*}} constant i32 6, align 4 #3

//CHECK: @n ={{.*}} global i32 0, align 4
//CHECK: @o ={{.*}} global i32 6, align 4
//CHECK: @p ={{.*}} constant i32 7, align 4
//CHECK: @_ZL5fptrs = internal constant [2 x i32 ()*] [i32 ()* @foo, i32 ()* @goo], align {{4|8}} #3

//CHECK: define{{.*}} i32 @foo() #5 {
//CHECK: define{{.*}} i32 @goo() #6 {
//CHECK: declare i32 @zoo(i32* noundef, i32* noundef) #7
//CHECK: define{{.*}} i32 @hoo() #8 {

//ELF: attributes #0 = { "bss-section"="my_bss.1" "data-section"="my_data.1" "rodata-section"="my_rodata.1" }
//ELF: attributes #1 = { "data-section"="my_data.1" "rodata-section"="my_rodata.1" }
//ELF: attributes #2 = { "bss-section"="my_bss.2" "rodata-section"="my_rodata.1" }
//ELF: attributes #3 = { "bss-section"="my_bss.2" "data-section"="my_data.2" "relro-section"="my_relro.2" "rodata-section"="my_rodata.2" }
//ELF: attributes #4 = { "relro-section"="my_relro.2" }
//ELF: attributes #5 = { {{.*"implicit-section-name"="my_text.1".*}} }
//ELF: attributes #6 = { {{.*"implicit-section-name"="my_text.2".*}} }
//MACHO: attributes #0 = { "bss-section"="__BSS,__mybss1" "data-section"="__DATA,__mydata1" "rodata-section"="__RODATA,__myrodata1" }
//MACHO: attributes #1 = { "data-section"="__DATA,__mydata1" "rodata-section"="__RODATA,__myrodata1" }
//MACHO: attributes #2 = { "bss-section"="__BSS,__mybss2" "rodata-section"="__RODATA,__myrodata1" }
//MACHO: attributes #3 = { "bss-section"="__BSS,__mybss2" "data-section"="__DATA,__mydata2" "relro-section"="__RELRO,__myrelro2" "rodata-section"="__RODATA,__myrodata2" }
//MACHO: attributes #4 = { "relro-section"="__RELRO,__myrelro2" }
//MACHO: attributes #5 = { {{.*"implicit-section-name"="__TEXT,__mytext1".*}} }
//MACHO: attributes #6 = { {{.*"implicit-section-name"="__TEXT,__mytext2".*}} }
//CHECK-NOT: attributes #7 = { {{.*"implicit-section-name".*}} }
//CHECK-NOT: attributes #8 = { {{.*"implicit-section-name".*}} }
