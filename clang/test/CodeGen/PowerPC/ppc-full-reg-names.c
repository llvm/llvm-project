// REQUIRES: powerpc-registered-target
// RUN: %clang -target powerpc-ibm-aix-xcoff -mcpu=pwr8 -O3 -S -emit-llvm -mregnames \
// RUN:   -maltivec %s -o - | FileCheck %s --check-prefix=FULLNAMES
// RUN: %clang -target powerpc64-ibm-aix-xcoff -mcpu=pwr8 -O3 -S -emit-llvm -mregnames \
// RUN:   -maltivec %s -o - | FileCheck %s --check-prefix=FULLNAMES
// RUN: %clang -target powerpc64le-unknown-linux-gnu -mcpu=pwr8 -O3 -S -emit-llvm -mregnames \
// RUN:   -maltivec %s -o - | FileCheck %s --check-prefix=FULLNAMES
// RUN: %clang -target powerpc-ibm-aix-xcoff -mcpu=pwr8 -O3 -S -emit-llvm -mno-regnames \
// RUN:   -maltivec %s -o - | FileCheck %s --check-prefix=NOFULLNAMES
// RUN: %clang -target powerpc64-ibm-aix-xcoff -mcpu=pwr8 -O3 -S -emit-llvm -mno-regnames \
// RUN:   -maltivec %s -o - | FileCheck %s --check-prefix=NOFULLNAMES
// RUN: %clang -target powerpc64le-unknown-linux-gnu -mcpu=pwr8 -O3 -S -emit-llvm -mno-regnames \
// RUN:   -maltivec %s -o - | FileCheck %s --check-prefix=NOFULLNAMES

// Also check the assembly to make sure that the full names are used.
// RUN: %clang -target powerpc-ibm-aix-xcoff -mcpu=pwr8 -O3 -S -mregnames \
// RUN:   -maltivec %s -o - | FileCheck %s --check-prefix=ASMFULLNAMES
// RUN: %clang -target powerpc64-ibm-aix-xcoff -mcpu=pwr8 -O3 -S -mregnames \
// RUN:   -maltivec %s -o - | FileCheck %s --check-prefix=ASMFULLNAMES
// RUN: %clang -target powerpc64le-unknown-linux-gnu -mcpu=pwr8 -O3 -S -mregnames \
// RUN:   -maltivec %s -o - | FileCheck %s --check-prefix=ASMFULLNAMES
// RUN: %clang -target powerpc-ibm-aix-xcoff -mcpu=pwr8 -O3 -S -mno-regnames \
// RUN:   -maltivec %s -o - | FileCheck %s --check-prefix=ASMNOFULLNAMES
// RUN: %clang -target powerpc64-ibm-aix-xcoff -mcpu=pwr8 -O3 -S -mno-regnames \
// RUN:   -maltivec %s -o - | FileCheck %s --check-prefix=ASMNOFULLNAMES
// RUN: %clang -target powerpc64le-unknown-linux-gnu -mcpu=pwr8 -O3 -S -mno-regnames \
// RUN:   -maltivec %s -o - | FileCheck %s --check-prefix=ASMNOFULLNAMES



// FULLNAMES-LABEL: @IntNames
// FULLNAMES-SAME:  #0
// NOFULLNAMES-LABEL: @IntNames
// NOFULLNAMES-SAME:  #0
// ASMFULLNAMES-LABEL: IntNames:
// ASMFULLNAMES:         add r3, r4, r3
// ASMFULLNAMES:         blr
// ASMNOFULLNAMES-LABEL: IntNames:
// ASMNOFULLNAMES:         add 3, 4, 3
// ASMNOFULLNAMES:         blr
int IntNames(int a, int b) {
  return a + b;
}

// FULLNAMES-LABEL: @FPNames
// FULLNAMES-SAME:  #0
// NOFULLNAMES-LABEL: @FPNames
// NOFULLNAMES-SAME:  #0
// ASMFULLNAMES-LABEL: FPNames:
// ASMFULLNAMES:         xsadddp f1, f1, f2
// ASMFULLNAMES:         blr
// ASMNOFULLNAMES-LABEL: FPNames:
// ASMNOFULLNAMES:         xsadddp 1, 1, 2
// ASMNOFULLNAMES:         blr
double FPNames(double a, double b) {
  return a + b;
}

// FULLNAMES-LABEL: @VecNames
// FULLNAMES-SAME:  #0
// NOFULLNAMES-LABEL: @VecNames
// NOFULLNAMES-SAME:  #0
// ASMFULLNAMES-LABEL: VecNames:
// ASMFULLNAMES:         xvaddsp vs34, vs34, vs35
// ASMFULLNAMES:         blr
// ASMNOFULLNAMES-LABEL: VecNames:
// ASMNOFULLNAMES:         xvaddsp 34, 34, 35
// ASMNOFULLNAMES:         blr
vector float VecNames(vector float a, vector float b) {
  return a + b;
}

// FULLNAMES: attributes #0 = {
// FULLNAMES-SAME: +regnames
// NOFULLNAMES: attributes #0 = {
// NOFULLNAMES-SAME: -regnames


