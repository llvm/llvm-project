// REQUIRES: arm-registered-target
// RUN: %clang -target arm-none-none-eabihf -mcpu=cortex-r5 -mfpu=vfpv3-d16 -marm -S -o - %s \
// RUN: | FileCheck %s --check-prefix=CHECK-R
// RUN: %clang -target arm-none-none-eabihf -mcpu=cortex-r5 -mfpu=vfpv3-d16 -mthumb -S -o - %s \
// RUN: | FileCheck %s --check-prefix=CHECK-R
// RUN: %clang -target arm-none-none-eabihf -mcpu=cortex-r4 -mfpu=vfpv3-d16 -marm -S -o - %s \
// RUN: | FileCheck %s --check-prefix=CHECK-R
// RUN: %clang -target arm-none-none-eabihf -mcpu=cortex-r4 -mfpu=vfpv3-d16 -mthumb -S -o - %s \
// RUN: | FileCheck %s --check-prefix=CHECK-R
// RUN: %clang -target arm-none-none-eabihf -mcpu=cortex-m4 -mfpu=fpv4-sp-d16 -S -o - %s \
// RUN: | FileCheck %s --check-prefix=CHECK-M
// RUN: %clang -target arm-none-none-eabihf -mcpu=cortex-m33 -mfpu=fpv5-sp-d16 -S -o - %s \
// RUN: | FileCheck %s --check-prefix=CHECK-M

void bar();

__attribute__((interrupt_save_fp)) void test_generic_interrupt() {
    // CHECK-R:      vmrs	r4, fpscr
    // CHECK-R-NEXT: vmrs	r5, fpexc
    // CHECK-R-NEXT: .save  {r4, r5}
    // CHECK-R-NEXT: push	{r4, r5}
    // .....
    // CHECK-R:      pop	{r4, r5}
    // CHECK-R-NEXT: vmsr	fpscr, r4
    // CHECK-R-NEXT: vmsr	fpexc, r5

    // CHECK-M:      vmrs	r4, fpscr
    // CHECK-M-NEXT: .save  {r4}
    // CHECK-M-NEXT: push	{r4}
    // .....
    // CHECK-M:      pop	{r4}
    // CHECK-M-NEXT: vmsr	fpscr, r4
    bar();
}
