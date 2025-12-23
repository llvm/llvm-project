// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O2 -disable-llvm-passes -frestrict-experimental -emit-llvm -o - %s | FileCheck %s

#include <stddef.h>

struct ComplexStruct {
    int id;
    float data;
    char *name;
    struct ComplexStruct *next;
};

// CHECK-LABEL: define dso_local void @test_deep_nesting_mixed(
// CHECK:         [[L1_INT:%.*]] = alloca ptr, align 8, !scope [[META2:![0-9]+]]
// CHECK-NEXT:    [[L1_FLOAT:%.*]] = alloca ptr, align 8, !scope [[META2]]
// CHECK-NEXT:    [[L1_STRUCT:%.*]] = alloca ptr, align 8, !scope [[META2]]
// CHECK-NEXT:    [[L2_INT:%.*]] = alloca ptr, align 8, !scope [[META4:![0-9]+]]
// CHECK-NEXT:    [[L2_FLOAT:%.*]] = alloca ptr, align 8, !scope [[META6:![0-9]+]]
// CHECK-NEXT:    [[L2_STRUCT:%.*]] = alloca ptr, align 8, !scope [[META8:![0-9]+]]
// CHECK-NEXT:    [[I:%.*]] = alloca i32, align 4
// CHECK-NEXT:    [[CLEANUP_DEST_SLOT:%.*]] = alloca i32, align 4
// CHECK-NEXT:    [[L3_DOUBLE:%.*]] = alloca ptr, align 8, !scope [[META10:![0-9]+]]
// CHECK-NEXT:    [[L3_LONG_RESTRICT:%.*]] = alloca ptr, align 8, !scope [[META11:![0-9]+]]
// CHECK-NEXT:    [[L3_COMPLEX:%.*]] = alloca ptr, align 8, !scope [[META10]]
// CHECK-NEXT:    [[L4_CHAR_RESTRICT:%.*]] = alloca ptr, align 8, !scope [[META13:![0-9]+]]
// CHECK-NEXT:    [[L4_USHORT:%.*]] = alloca ptr, align 8, !scope [[META15:![0-9]+]]
// CHECK-NEXT:    [[L4_STRUCT_RESTRICT:%.*]] = alloca ptr, align 8, !scope [[META16:![0-9]+]]
// CHECK-NEXT:    [[L5_INT_RESTRICT:%.*]] = alloca ptr, align 8, !scope [[META18:![0-9]+]]
// CHECK-NEXT:    [[L5_FLOAT:%.*]] = alloca ptr, align 8, !scope [[META20:![0-9]+]]
//
void test_deep_nesting_mixed() {
    int *l1_int;
    float *l1_float;
    struct ComplexStruct *l1_struct;


    if (1) {
        int * restrict l2_int;
        float * restrict l2_float;
        struct ComplexStruct * restrict l2_struct;

        for (int i = 0; i < 5; i++) {
            double *l3_double;
            long * restrict l3_long_restrict;
            struct ComplexStruct *l3_complex;
            {
                char * restrict l4_char_restrict;
                unsigned short *l4_ushort;
                struct ComplexStruct * restrict l4_struct_restrict;

                switch (i) {
                    case 0: {
                        int * restrict l5_int_restrict;
                        break;
                    }
                    default: {
                        float *l5_float;
                    }
                }
            }
        }
    }
}

// CHECK-LABEL: define dso_local void @test_same_name_different_types(
// CHECK:         [[PTR:%.*]] = alloca ptr, align 8, !scope [[META27:![0-9]+]]
// CHECK-NEXT:    [[PTR1:%.*]] = alloca ptr, align 8, !scope [[META28:![0-9]+]]
// CHECK-NEXT:    [[PTR2:%.*]] = alloca ptr, align 8, !scope [[META29:![0-9]+]]
// CHECK-NEXT:    [[PTR3:%.*]] = alloca ptr, align 8, !scope [[META30:![0-9]+]]
// CHECK-NEXT:    [[PTR4:%.*]] = alloca ptr, align 8, !scope [[META31:![0-9]+]]
// CHECK-NEXT:    [[PTR5:%.*]] = alloca ptr, align 8, !scope [[META32:![0-9]+]]
//
void test_same_name_different_types() {
    int *ptr;
    {
        float * restrict ptr;

        {
            char *ptr;
        }
    }

    {
        struct { int val; } * restrict ptr;
    }

    {
        double *ptr;
        {
            long * restrict ptr;
        }
    }
}

//.
// CHECK: [[META2]] = !{!"test_deep_nesting_mixed_1", [[META3:![0-9]+]]}
// CHECK: [[META3]] = !{i64 0}
// CHECK: [[META4]] = !{!"test_deep_nesting_mixed_10", [[META5:![0-9]+]]}
// CHECK: [[META5]] = !{i64 1}
// CHECK: [[META6]] = !{!"test_deep_nesting_mixed_10", [[META7:![0-9]+]]}
// CHECK: [[META7]] = !{i64 2}
// CHECK: [[META8]] = !{!"test_deep_nesting_mixed_10", [[META9:![0-9]+]]}
// CHECK: [[META9]] = !{i64 3}
// CHECK: [[META10]] = !{!"test_deep_nesting_mixed_100", [[META3]]}
// CHECK: [[META11]] = !{!"test_deep_nesting_mixed_100", [[META12:![0-9]+]]}
// CHECK: [[META12]] = !{i64 4}
// CHECK: [[META13]] = !{!"test_deep_nesting_mixed_1000", [[META14:![0-9]+]]}
// CHECK: [[META14]] = !{i64 5}
// CHECK: [[META15]] = !{!"test_deep_nesting_mixed_1000", [[META3]]}
// CHECK: [[META16]] = !{!"test_deep_nesting_mixed_1000", [[META17:![0-9]+]]}
// CHECK: [[META17]] = !{i64 6}
// CHECK: [[META18]] = !{!"test_deep_nesting_mixed_100000", [[META19:![0-9]+]]}
// CHECK: [[META19]] = !{i64 7}
// CHECK: [[META20]] = !{!"test_deep_nesting_mixed_1000001", [[META3]]}
// CHECK: [[META27]] = !{!"test_same_name_different_types_1", [[META3]]}
// CHECK: [[META28]] = !{!"test_same_name_different_types_10", [[META5]]}
// CHECK: [[META29]] = !{!"test_same_name_different_types_100", [[META3]]}
// CHECK: [[META30]] = !{!"test_same_name_different_types_101", [[META7]]}
// CHECK: [[META31]] = !{!"test_same_name_different_types_1011", [[META3]]}
// CHECK: [[META32]] = !{!"test_same_name_different_types_10110", [[META9]]}
//.
