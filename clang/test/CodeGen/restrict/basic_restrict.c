// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O2 -disable-llvm-passes -frestrict-experimental -emit-llvm -o - %s | FileCheck %s

#include <stdalign.h>

// CHECK-LABEL: define dso_local void @test_basic_restrict(
// CHECK:         [[P1_ADDR:%.*]] = alloca ptr, align 8, !scope [[META2:![0-9]+]]
// CHECK-NEXT:    [[P2_ADDR:%.*]] = alloca ptr, align 8, !scope [[META4:![0-9]+]]
// CHECK-NEXT:    [[X_ADDR:%.*]] = alloca ptr, align 8, !scope [[META6:![0-9]+]]
//
void test_basic_restrict(int * restrict p1, int * restrict p2, int x[restrict 5]) {
}

// CHECK-LABEL: define dso_local void @test_nested_restrict(
// CHECK:         [[A:%.*]] = alloca ptr, align 8, !scope [[META13:![0-9]+]]
// CHECK-NEXT:    [[B:%.*]] = alloca ptr, align 8, !scope [[META13]]
// CHECK-NEXT:    [[C:%.*]] = alloca ptr, align 8, !scope [[META13]]
// CHECK-NEXT:    [[B_INT:%.*]] = alloca ptr, align 8, !scope [[META15:![0-9]+]]
// CHECK-NEXT:    [[B_FLOAT:%.*]] = alloca ptr, align 8, !scope [[META16:![0-9]+]]
// CHECK-NEXT:    [[I:%.*]] = alloca i32, align 4
// CHECK-NEXT:    [[C_DOUBLE:%.*]] = alloca ptr, align 8, !scope [[META17:![0-9]+]]
// CHECK-NEXT:    [[C_LONG:%.*]] = alloca ptr, align 8, !scope [[META18:![0-9]+]]
// CHECK-NEXT:    [[D_SHORT:%.*]] = alloca ptr, align 8, !scope [[META20:![0-9]+]]
// CHECK-NEXT:    [[D_UNSIGNED:%.*]] = alloca ptr, align 8, !scope [[META22:![0-9]+]]
// CHECK-NEXT:    [[E_STRUCT:%.*]] = alloca ptr, align 8, !scope [[META24:![0-9]+]]
//
void test_nested_restrict() {
  int *a;
  float *b;
  char *c;

  if (1) {
    int * restrict b_int;
    float * restrict b_float;
  }

  for (int i = 0; i < 10; i++) {
    double * restrict c_double;
    long * restrict c_long;
  }

  while (1) {
    short * restrict d_short;
    unsigned * restrict d_unsigned;
    break;
  }

  {
    struct Point {
      int x;
      int y;
    } * restrict e_struct;
  }
}

// CHECK-LABEL: define dso_local void @test_assignment(
// CHECK:         [[R1_INT:%.*]] = alloca ptr, align 8, !scope [[META30:![0-9]+]]
// CHECK-NEXT:    [[R2_FLOAT:%.*]] = alloca ptr, align 8, !scope [[META31:![0-9]+]]
// CHECK-NEXT:    [[NORMAL_INT:%.*]] = alloca ptr, align 8, !scope [[META32:![0-9]+]]
// CHECK-NEXT:    [[NORMAL_FLOAT:%.*]] = alloca ptr, align 8, !scope [[META32]]
// CHECK-NEXT:    [[R3_INT:%.*]] = alloca ptr, align 8, !scope [[META33:![0-9]+]]
// CHECK-NEXT:    [[R4_FLOAT:%.*]] = alloca ptr, align 8, !scope [[META34:![0-9]+]]
//
void test_assignment() {
  int * restrict r1_int;
  float * restrict r2_float;

  {
    int *normal_int;
    float *normal_float;
    normal_int = r1_int;
    normal_float = r2_float;
  }

  {
    int * restrict r3_int;
    float * restrict r4_float;
    r3_int = r1_int;
    r4_float = r2_float;
  }
}

// CHECK-LABEL: define dso_local void @test_mixed_qualifiers(
// CHECK:         [[CONST_RESTRICT_INT:%.*]] = alloca ptr, align 8, !scope [[META37:![0-9]+]]
// CHECK-NEXT:    [[CONST_RESTRICT_STRUCT:%.*]] = alloca ptr, align 8, !scope [[META38:![0-9]+]]
// CHECK-NEXT:    [[VOLATILE_RESTRICT_INT:%.*]] = alloca ptr, align 8, !scope [[META39:![0-9]+]]
// CHECK-NEXT:    [[CONST_VOLATILE_RESTRICT_INT:%.*]] = alloca ptr, align 8, !scope [[META40:![0-9]+]]
// CHECK-NEXT:    [[VALUE:%.*]] = alloca i32, align 4
// CHECK-NEXT:    [[LOCAL_CONST_RESTRICT_FLOAT:%.*]] = alloca ptr, align 8, !scope [[META41:![0-9]+]]
// CHECK-NEXT:    [[LOCAL_VOLATILE_RESTRICT_CHAR:%.*]] = alloca ptr, align 8, !scope [[META42:![0-9]+]]
//
void test_mixed_qualifiers() {
    const int * restrict const_restrict_int;
    const struct { int a; float b; } * restrict const_restrict_struct;

    volatile int * restrict volatile_restrict_int;

    const volatile int * restrict const_volatile_restrict_int;

    int value = 42;
    const_restrict_int = &value;

    {
        const float * restrict local_const_restrict_float;
        volatile char * restrict local_volatile_restrict_char;
    }
}

// CHECK-LABEL: define dso_local void @test_aligned_types(
// CHECK:         [[ALIGNED_PTR:%.*]] = alloca ptr, align 16, !scope [[META43:![0-9]+]]
// CHECK-NEXT:    [[ALIGNED_RESTRICT_PTR:%.*]] = alloca ptr, align 16, !scope [[META44:![0-9]+]]
// CHECK-NEXT:    [[ALIGNED_STRUCT_RESTRICT:%.*]] = alloca ptr, align 64, !scope [[META45:![0-9]+]]
//
void test_aligned_types() {
    alignas(16) int *aligned_ptr;
    alignas(16) int * restrict aligned_restrict_ptr;

    {
        alignas(64) struct {
            int a;
            double b;
        } * restrict aligned_struct_restrict;

    }
}

// CHECK-LABEL: define dso_local void @test_loops(
// CHECK:         [[I:%.*]] = alloca i32, align 4
// CHECK-NEXT:    [[P:%.*]] = alloca ptr, align 8, !scope [[META46:![0-9]+]]
// CHECK-NEXT:    [[Q:%.*]] = alloca ptr, align 8, !scope [[META47:![0-9]+]]
// CHECK-NEXT:    [[R:%.*]] = alloca ptr, align 8, !scope [[META48:![0-9]+]]
// CHECK-NEXT:    [[COUNTER:%.*]] = alloca i32, align 4
// CHECK-NEXT:    [[W:%.*]] = alloca ptr, align 8, !scope [[META49:![0-9]+]]
// CHECK-NEXT:    [[INNER:%.*]] = alloca ptr, align 8, !scope [[META50:![0-9]+]]
// CHECK-NEXT:    [[D:%.*]] = alloca ptr, align 8, !scope [[META51:![0-9]+]]
//
void test_loops() {
  for (int i = 0; i < 3; i++) {
    int * restrict p;

    if (i % 2 == 0) {
      int *q;
    } else {
      int * restrict r;
    }
  }

  int counter = 0;
  while (counter < 2) {
    int * restrict w;

    {
      int *inner;
    }
    counter++;
  }

  do {
    int * restrict d;
    counter--;
  } while (counter > 0);
}

// CHECK-LABEL: define dso_local void @test_restrict_complex_pointers(
// CHECK:         [[BASE:%.*]] = alloca ptr, align 8, !scope [[META55:![0-9]+]]
// CHECK-NEXT:    [[NOT_POINTER:%.*]] = alloca i32, align 4
// CHECK-NEXT:    [[IF_RESTRICT:%.*]] = alloca ptr, align 8, !scope [[META57:![0-9]+]]
// CHECK-NEXT:    [[IF_TRIPLE:%.*]] = alloca ptr, align 8, !scope [[META59:![0-9]+]]
// CHECK-NEXT:    [[I:%.*]] = alloca i32, align 4
// CHECK-NEXT:    [[CLEANUP_DEST_SLOT:%.*]] = alloca i32, align 4
// CHECK-NEXT:    [[LOOP_RESTRICT:%.*]] = alloca ptr, align 8, !scope [[META61:![0-9]+]]
// CHECK-NEXT:    [[LOOP_QUAD:%.*]] = alloca ptr, align 8, !scope [[META63:![0-9]+]]
// CHECK-NEXT:    [[CASE_RESTRICT:%.*]] = alloca ptr, align 8, !scope [[META65:![0-9]+]]
// CHECK-NEXT:    [[CASE_TRIPLE:%.*]] = alloca ptr, align 8, !scope [[META67:![0-9]+]]
// CHECK-NEXT:    [[CASE_QUAD:%.*]] = alloca ptr, align 8, !scope [[META69:![0-9]+]]
// CHECK-NEXT:    [[BLOCK_RESTRICT:%.*]] = alloca ptr, align 8, !scope [[META71:![0-9]+]]
// CHECK-NEXT:    [[BLOCK_TRIPLE:%.*]] = alloca ptr, align 8, !scope [[META73:![0-9]+]]
// CHECK-NEXT:    [[NESTED_RESTRICT:%.*]] = alloca ptr, align 8, !scope [[META75:![0-9]+]]
// CHECK-NEXT:    [[NESTED_QUAD:%.*]] = alloca ptr, align 8, !scope [[META77:![0-9]+]]
//
void test_restrict_complex_pointers() {
    int ** base;
    int not_pointer;

    if (1) {
        int * restrict * restrict if_restrict;
        int *** restrict if_triple;
    }

    for (int i = 0; i < 10; i++) {
        int * restrict * restrict loop_restrict;
        int **** restrict loop_quad;

        switch (i % 3) {
            case 0: {
                int * restrict * restrict case_restrict;
                break;
            }
            case 1: {
                int *** restrict case_triple;
                break;
            }
            case 2: {
                int * restrict * restrict * case_quad;
                break;
            }
        }
    }

    {
        int * restrict * restrict block_restrict;
        int *** restrict block_triple;

        {
            int * restrict * restrict * nested_restrict;
            int **** restrict nested_quad;
        }
    }
}

//.
// CHECK: [[META2]] = !{!"test_basic_restrict_1", [[META3:![0-9]+]]}
// CHECK: [[META3]] = !{i64 1}
// CHECK: [[META4]] = !{!"test_basic_restrict_1", [[META5:![0-9]+]]}
// CHECK: [[META5]] = !{i64 2}
// CHECK: [[META6]] = !{!"test_basic_restrict_1", [[META7:![0-9]+]]}
// CHECK: [[META7]] = !{i64 3}
// CHECK: [[META13]] = !{!"test_nested_restrict_1", [[META14:![0-9]+]]}
// CHECK: [[META14]] = !{i64 0}
// CHECK: [[META15]] = !{!"test_nested_restrict_10", [[META3]]}
// CHECK: [[META16]] = !{!"test_nested_restrict_10", [[META5]]}
// CHECK: [[META17]] = !{!"test_nested_restrict_101", [[META7]]}
// CHECK: [[META18]] = !{!"test_nested_restrict_101", [[META19:![0-9]+]]}
// CHECK: [[META19]] = !{i64 4}
// CHECK: [[META20]] = !{!"test_nested_restrict_1011", [[META21:![0-9]+]]}
// CHECK: [[META21]] = !{i64 5}
// CHECK: [[META22]] = !{!"test_nested_restrict_1011", [[META23:![0-9]+]]}
// CHECK: [[META23]] = !{i64 6}
// CHECK: [[META24]] = !{!"test_nested_restrict_10111", [[META25:![0-9]+]]}
// CHECK: [[META25]] = !{i64 7}
// CHECK: [[META30]] = !{!"test_assignment_1", [[META3]]}
// CHECK: [[META31]] = !{!"test_assignment_1", [[META5]]}
// CHECK: [[META32]] = !{!"test_assignment_10", [[META14]]}
// CHECK: [[META33]] = !{!"test_assignment_101", [[META7]]}
// CHECK: [[META34]] = !{!"test_assignment_101", [[META19]]}
// CHECK: [[META37]] = !{!"test_mixed_qualifiers_1", [[META3]]}
// CHECK: [[META38]] = !{!"test_mixed_qualifiers_1", [[META5]]}
// CHECK: [[META39]] = !{!"test_mixed_qualifiers_1", [[META7]]}
// CHECK: [[META40]] = !{!"test_mixed_qualifiers_1", [[META19]]}
// CHECK: [[META41]] = !{!"test_mixed_qualifiers_10", [[META21]]}
// CHECK: [[META42]] = !{!"test_mixed_qualifiers_10", [[META23]]}
// CHECK: [[META43]] = !{!"test_aligned_types_1", [[META14]]}
// CHECK: [[META44]] = !{!"test_aligned_types_1", [[META3]]}
// CHECK: [[META45]] = !{!"test_aligned_types_10", [[META5]]}
// CHECK: [[META46]] = !{!"test_loops_10", [[META3]]}
// CHECK: [[META47]] = !{!"test_loops_100", [[META14]]}
// CHECK: [[META48]] = !{!"test_loops_1001", [[META5]]}
// CHECK: [[META49]] = !{!"test_loops_101", [[META7]]}
// CHECK: [[META50]] = !{!"test_loops_1010", [[META14]]}
// CHECK: [[META51]] = !{!"test_loops_1011", [[META19]]}
// CHECK: [[META55]] = !{!"test_restrict_complex_pointers_1", [[META56:![0-9]+]]}
// CHECK: [[META56]] = !{i64 0, i64 0}
// CHECK: [[META57]] = !{!"test_restrict_complex_pointers_10", [[META58:![0-9]+]]}
// CHECK: [[META58]] = !{i64 1, i64 1}
// CHECK: [[META59]] = !{!"test_restrict_complex_pointers_10", [[META60:![0-9]+]]}
// CHECK: [[META60]] = !{i64 2, i64 0, i64 0}
// CHECK: [[META61]] = !{!"test_restrict_complex_pointers_101", [[META62:![0-9]+]]}
// CHECK: [[META62]] = !{i64 3, i64 2}
// CHECK: [[META63]] = !{!"test_restrict_complex_pointers_101", [[META64:![0-9]+]]}
// CHECK: [[META64]] = !{i64 4, i64 0, i64 0, i64 0}
// CHECK: [[META65]] = !{!"test_restrict_complex_pointers_10100", [[META66:![0-9]+]]}
// CHECK: [[META66]] = !{i64 5, i64 3}
// CHECK: [[META67]] = !{!"test_restrict_complex_pointers_101001", [[META68:![0-9]+]]}
// CHECK: [[META68]] = !{i64 6, i64 0, i64 0}
// CHECK: [[META69]] = !{!"test_restrict_complex_pointers_1010011", [[META70:![0-9]+]]}
// CHECK: [[META70]] = !{i64 0, i64 4, i64 1}
// CHECK: [[META71]] = !{!"test_restrict_complex_pointers_1011", [[META72:![0-9]+]]}
// CHECK: [[META72]] = !{i64 7, i64 5}
// CHECK: [[META73]] = !{!"test_restrict_complex_pointers_1011", [[META74:![0-9]+]]}
// CHECK: [[META74]] = !{i64 8, i64 0, i64 0}
// CHECK: [[META75]] = !{!"test_restrict_complex_pointers_10110", [[META76:![0-9]+]]}
// CHECK: [[META76]] = !{i64 0, i64 6, i64 2}
// CHECK: [[META77]] = !{!"test_restrict_complex_pointers_10110", [[META78:![0-9]+]]}
// CHECK: [[META78]] = !{i64 9, i64 0, i64 0, i64 0}
//.
