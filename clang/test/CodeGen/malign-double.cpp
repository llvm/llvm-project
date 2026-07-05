// RUN: %clang_cc1 -malign-double -triple i386-unknown-linux -emit-llvm %s -o - \
// RUN:   | FileCheck --check-prefix=CHECK-ON --check-prefix=CHECK %s

// RUN: %clang_cc1 -triple i386-unknown-linux -emit-llvm %s -o - \
// RUN:   | FileCheck --check-prefix=CHECK-OFF --check-prefix=CHECK %s

/* Structs S1, S2, S3, S4, and union U5 are taken from Intel, "IA-64
   Software Conventions and Runtime Architecture Guide", version of
   August 1999, Section 4.2, Figures 4-1 through 4-5.
   A Union containing a double was also thrown in for good measure. */

struct S1 {
  char c;
};

unsigned S1_align = __alignof(struct S1);
unsigned S1_size = sizeof(struct S1);
// CHECK: @S1_align ={{.*}} global i32 1, align 4
// CHECK: @S1_size ={{.*}} global i32 1, align 4

unsigned S1_c_offset = (unsigned) &((struct S1*) 0)->c;
// CHECK: @S1_c_offset ={{.*}} global i32 0, align 4

struct S2{
  char c;
  char d;
  short s;
  int n;
};

unsigned S2_align = __alignof(struct S2);
unsigned S2_size = sizeof(struct S2);
// CHECK: @S2_align ={{.*}} global i32 4, align 4
// CHECK: @S2_size ={{.*}} global i32 8, align 4

unsigned S2_c_offset = (unsigned) &((struct S2*) 0)->c;
unsigned S2_d_offset = (unsigned) &((struct S2*) 0)->d;
unsigned S2_s_offset = (unsigned) &((struct S2*) 0)->s;
unsigned S2_n_offset = (unsigned) &((struct S2*) 0)->n;
// CHECK: @S2_c_offset ={{.*}} global i32 0, align 4
// CHECK: @S2_d_offset ={{.*}} global i32 1, align 4
// CHECK: @S2_s_offset ={{.*}} global i32 2, align 4
// CHECK: @S2_n_offset ={{.*}} global i32 4, align 4

struct S3 {
  char c;
  short s;
};

unsigned S3_align = __alignof(struct S3);
unsigned S3_size = sizeof(struct S3);
// CHECK: @S3_align ={{.*}} global i32 2, align 4
// CHECK: @S3_size ={{.*}} global i32 4, align 4

unsigned S3_c_offset = (unsigned) &((struct S3*) 0)->c;
unsigned S3_s_offset = (unsigned) &((struct S3*) 0)->s;
// CHECK: @S3_c_offset ={{.*}} global i32 0, align 4
// CHECK: @S3_s_offset ={{.*}} global i32 2, align 4

struct S4 {
  char c;
  double d;
  short s;
};

unsigned S4_align = __alignof(struct S4);
unsigned S4_size = sizeof(struct S4);
// CHECK-ON: @S4_align ={{.*}} global i32 8, align 4
// CHECK-ON: @S4_size ={{.*}} global i32 24, align 4
// CHECK-OFF: @S4_align ={{.*}} global i32 4, align 4
// CHECK-OFF: @S4_size ={{.*}} global i32 16, align 4

unsigned S4_c_offset = (unsigned) &((struct S4*) 0)->c;
unsigned S4_d_offset = (unsigned) &((struct S4*) 0)->d;
unsigned S4_s_offset = (unsigned) &((struct S4*) 0)->s;
// CHECK: @S4_c_offset ={{.*}} global i32 0, align 4
// CHECK-ON: @S4_d_offset ={{.*}} global i32 8, align 4
// CHECK-ON: @S4_s_offset ={{.*}} global i32 16, align 4
// CHECK-OFF: @S4_d_offset ={{.*}} global i32 4, align 4
// CHECK-OFF: @S4_s_offset ={{.*}} global i32 12, align 4

union S5 {
  char c;
  short s;
  int j;
};

unsigned S5_align = __alignof(union S5);
unsigned S5_size = sizeof(union S5);
// CHECK: @S5_align ={{.*}} global i32 4, align 4
// CHECK: @S5_size ={{.*}} global i32 4, align 4

unsigned S5_c_offset = (unsigned) &((union S5*) 0)->c;
unsigned S5_s_offset = (unsigned) &((union S5*) 0)->s;
unsigned S5_j_offset = (unsigned) &((union S5*) 0)->j;
// CHECK: @S5_c_offset ={{.*}} global i32 0, align 4
// CHECK: @S5_s_offset ={{.*}} global i32 0, align 4
// CHECK: @S5_j_offset ={{.*}} global i32 0, align 4

union S6 {
  char c;
  double d;
};

unsigned S6_align = __alignof(union S6);
unsigned S6_size = sizeof(union S6);
// CHECK-ON: @S6_align ={{.*}} global i32 8, align 4
// CHECK-ON: @S6_size ={{.*}} global i32 8, align 4
// CHECK-OFF: @S6_align ={{.*}} global i32 4, align 4
// CHECK-OFF: @S6_size ={{.*}} global i32 8, align 4

unsigned S6_c_offset = (unsigned) &((union S6*) 0)->c;
unsigned S6_d_offset = (unsigned) &((union S6*) 0)->d;
// CHECK: @S6_c_offset ={{.*}} global i32 0, align 4
// CHECK: @S6_d_offset ={{.*}} global i32 0, align 4
