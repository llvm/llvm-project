// RUN: %clang_cc1 %s -triple mips -target-feature +soft-float \
// RUN:            -DSOFT_FLOAT_CONSTRAINT_R \
// RUN:            -DFLOAT=float -emit-llvm -o - \
// RUN:      | FileCheck %s --check-prefix SOFT_FLOAT_CONSTRAINT_R_SINGLE

// RUN: %clang_cc1 %s -triple mips -target-feature +soft-float \
// RUN:            -DSOFT_FLOAT_CONSTRAINT_R \
// RUN:            -DFLOAT=double -emit-llvm -o - \
// RUN:      | FileCheck %s --check-prefix SOFT_FLOAT_CONSTRAINT_R_DOUBLE

#ifdef SOFT_FLOAT_CONSTRAINT_R
// SOFT_FLOAT_CONSTRAINT_R_SINGLE: call void asm sideeffect "", "r,~{$1}"(float %2) #1, !srcloc !2
// SOFT_FLOAT_CONSTRAINT_R_DOUBLE: call void asm sideeffect "", "r,~{$1}"(double %2) #1, !srcloc !2
void read_float(FLOAT* p) {
    FLOAT result = *p;
    __asm__("" ::"r"(result));
}
#endif // SOFT_FLOAT_CONSTRAINT_R
