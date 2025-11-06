// RUN: %clang_cc1 -triple s390x-linux-gnu %s -o - -emit-llvm \
// RUN:    | FileCheck %s -check-prefixes=CHECK,ALIGNED

// RUN: %clang_cc1 -triple s390x-linux-gnu %s -o - -emit-llvm \
// RUN:    -target-feature -unaligned-symbols | FileCheck %s -check-prefixes=CHECK,ALIGNED

// RUN: %clang_cc1 -triple s390x-linux-gnu %s -o - -emit-llvm \
// RUN:    -target-feature +unaligned-symbols | FileCheck %s -check-prefixes=CHECK,UNALIGN


// With -munaligned-symbols, the external and unaligned ("ExtUnal...")
// variable of each test should be treated as unaligned. For the explicitly
// aligned ("ExtExplAlign...") variables and those defined in the translation
// unit ("Aligned..."), the normal ABI alignment of 2 should still be
// in effect.

// ALIGNED: @ExtUnal = external global i8, align 2
// UNALIGN: @ExtUnal = external global i8, align 1
// CHECK:   @ExtExplAlign = external global i8, align 2
// CHECK:   @Aligned = {{(dso_local )?}}global i8 0, align 2
extern unsigned char ExtUnal;
extern unsigned char ExtExplAlign __attribute__((aligned(2)));
unsigned char Aligned;
unsigned char foo0 () {
  return ExtUnal + ExtExplAlign + Aligned;
}

// ALIGNED: @ExtUnal_c2Arr = external global [2 x i8], align 2
// UNALIGN: @ExtUnal_c2Arr = external global [2 x i8], align 1
// CHECK:   @ExtExplAlign_c2Arr = external global [2 x i8], align 2
// CHECK:   @Aligned_c2Arr = {{(dso_local )?}}global [2 x i8] zeroinitializer, align 2
extern unsigned char ExtUnal_c2Arr[2];
extern unsigned char ExtExplAlign_c2Arr[2] __attribute__((aligned(2)));
unsigned char Aligned_c2Arr[2];
unsigned char foo1 () {
  return ExtUnal_c2Arr[0] + ExtExplAlign_c2Arr[0] + Aligned_c2Arr[0];
}

// ALIGNED: @ExtUnal_s1c = external global %struct.s1c, align 2
// UNALIGN: @ExtUnal_s1c = external global %struct.s1c, align 1
// CHECK:   @ExtExplAlign_s1c = external global %struct.s1c, align 2
// CHECK:   @Aligned_s1c = {{(dso_local )?}}global %struct.s1c zeroinitializer, align 2
struct s1c { char c; };
extern struct s1c ExtUnal_s1c;
extern struct s1c ExtExplAlign_s1c __attribute__((aligned(2)));
struct s1c Aligned_s1c;
unsigned char foo2 () {
  return ExtUnal_s1c.c + ExtExplAlign_s1c.c + Aligned_s1c.c;
}

// ALIGNED: @ExtUnal_s2c = external global %struct.s2c, align 2
// UNALIGN: @ExtUnal_s2c = external global %struct.s2c, align 1
// CHECK:   @ExtExplAlign_s2c = external global %struct.s2c, align 2
// CHECK:   @Aligned_s2c = {{(dso_local )?}}global %struct.s2c zeroinitializer, align 2
struct s2c { char c; char c1;};
extern struct s2c ExtUnal_s2c;
extern struct s2c ExtExplAlign_s2c __attribute__((aligned(2)));
struct s2c Aligned_s2c;
unsigned char foo3 () {
  return ExtUnal_s2c.c + ExtExplAlign_s2c.c + Aligned_s2c.c;
}

// ALIGNED: @ExtUnal_s_c2Arr = external global %struct.s_c2Arr, align 2
// UNALIGN: @ExtUnal_s_c2Arr = external global %struct.s_c2Arr, align 1
// CHECK:   @ExtExplAlign_s_c2Arr = external global %struct.s_c2Arr, align 2
// CHECK:   @Aligned_s_c2Arr = {{(dso_local )?}}global %struct.s_c2Arr zeroinitializer, align 2
struct s_c2Arr { char c[2]; };
extern struct s_c2Arr ExtUnal_s_c2Arr;
extern struct s_c2Arr ExtExplAlign_s_c2Arr __attribute__((aligned(2)));
struct s_c2Arr Aligned_s_c2Arr;
unsigned char foo4 () {
  return ExtUnal_s_c2Arr.c[0] + ExtExplAlign_s_c2Arr.c[0] + Aligned_s_c2Arr.c[0];
}

// ALIGNED: @ExtUnal_s_packed = external global %struct.s_packed, align 2
// UNALIGN: @ExtUnal_s_packed = external global %struct.s_packed, align 1
// CHECK:   @ExtExplAlign_s_packed = external global %struct.s_packed, align 2
// CHECK:   @Aligned_s_packed = {{(dso_local )?}}global %struct.s_packed zeroinitializer, align 2
struct s_packed {
    int __attribute__((__packed__)) i;
    char c;
};
extern struct s_packed ExtUnal_s_packed;
extern struct s_packed ExtExplAlign_s_packed __attribute__((aligned(2)));
struct s_packed Aligned_s_packed;
unsigned char foo5 () {
  return ExtUnal_s_packed.c + ExtExplAlign_s_packed.c + Aligned_s_packed.c;
}

// ALIGNED: @ExtUnAl_s_nested = external global [2 x %struct.s_nested], align 2
// UNALIGN: @ExtUnAl_s_nested = external global [2 x %struct.s_nested], align 1
// CHECK:   @ExtExplAlign_s_nested = external global [2 x %struct.s_nested], align 2
// CHECK:   @Aligned_s_nested = {{(dso_local )?}}global [2 x %struct.s_nested] zeroinitializer, align 2
struct s_nested { struct s_c2Arr a[2]; };
extern struct s_nested ExtUnAl_s_nested[2];
extern struct s_nested ExtExplAlign_s_nested[2] __attribute__((aligned(2)));
struct s_nested Aligned_s_nested[2];
unsigned char foo6 () {
  return ExtUnAl_s_nested[0].a[0].c[0] + ExtExplAlign_s_nested[0].a[0].c[0] +
         Aligned_s_nested[0].a[0].c[0];
}

// A weak symbol could be replaced with an unaligned one at link time.
// CHECK-LABEL: foo7
// ALIGNED: load i8, ptr @Weaksym, align 2
// UNALIGN: load i8, ptr @Weaksym, align 1
unsigned char __attribute__((weak)) Weaksym = 0;
unsigned char foo7 () {
  return Weaksym;
}
