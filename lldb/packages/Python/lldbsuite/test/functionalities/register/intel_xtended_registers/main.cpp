//===-- main.cpp ------------------------------------------------*- C++ -*-===//
////
////                     The LLVM Compiler Infrastructure
////
//// This file is distributed under the University of Illinois Open Source
//// License. See LICENSE.TXT for details.
////
////===----------------------------------------------------------------------===//
//

#include <cpuid.h>
#include <cstddef>

int
main(int argc, char const *argv[])
{
    // This call returns 0 only if the CPU and the kernel support Intel(R) MPX.
    if (prctl(PR_MPX_ENABLE_MANAGEMENT, 0, 0, 0, 0) != 0)
        return -1;

// Run Intel(R) MPX test code.
#if defined(__x86_64__)
    asm("mov $16, %rax\n\t"
        "mov $9, %rdx\n\t"
        "bndmk (%rax,%rdx), %bnd0\n\t"
        "mov $32, %rax\n\t"
        "mov $9, %rdx\n\t"
        "bndmk (%rax,%rdx), %bnd1\n\t"
        "mov $48, %rax\n\t"
        "mov $9, %rdx\n\t"
        "bndmk (%rax,%rdx), %bnd2\n\t"
        "mov $64, %rax\n\t"
        "mov $9, %rdx\n\t"
        "bndmk (%rax,%rdx), %bnd3\n\t"
        "bndstx %bnd3, (%rax) \n\t"
        "nop\n\t");
#endif
#if defined(__i386__)
    asm("mov $16, %eax\n\t"
        "mov $9, %edx\n\t"
        "bndmk (%eax,%edx), %bnd0\n\t"
        "mov $32, %eax\n\t"
        "mov $9, %edx\n\t"
        "bndmk (%eax,%edx), %bnd1\n\t"
        "mov $48, %eax\n\t"
        "mov $9, %edx\n\t"
        "bndmk (%eax,%edx), %bnd2\n\t"
        "mov $64, %eax\n\t"
        "mov $9, %edx\n\t"
        "bndmk (%eax,%edx), %bnd3\n\t"
        "bndstx  %bnd3, (%eax)\n\t"
        "nop\n\t");
#endif
    asm("nop\n\t"); // Set a break point here.

    return 0;
}
