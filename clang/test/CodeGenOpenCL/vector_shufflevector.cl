// RUN: %clang_cc1 -emit-llvm -O0 %s -o - | FileCheck %s

// The shuffle vector mask must always be of i32 vector type
// See http://reviews.llvm.org/D10838 and https://llvm.org/bugs/show_bug.cgi?id=23800#c2
// for more information about a bug where a 64 bit index operand causes the generation
// of an invalid mask

typedef unsigned int uint2 __attribute((ext_vector_type(2)));

void vector_shufflevector_valid(void) {
    //CHECK: {{%.*}} = shufflevector <2 x i32> {{%.*}}, <2 x i32> undef, <2 x i32> <i32 0, i32 poison>
    (uint2)(((uint2)(0)).s0, 0);
}
