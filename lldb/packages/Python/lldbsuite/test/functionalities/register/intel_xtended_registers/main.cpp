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
    unsigned int rax, rbx, rcx, rdx;

    // Check if XSAVE is enabled.
    if (!__get_cpuid(1, &rax, &rbx, &rcx, &rdx) || (rcx & bit_OSXSAVE) != bit_OSXSAVE)
        return -1;

    // Check if MPX is enabled.
    if (__get_cpuid_max(0, NULL) > 7)
    {
        __cpuid_count(7, 0, rax, rbx, rcx, rdx);
        if ((rbx & bit_MPX) != bit_MPX)
            return -1;
    }
    else
        return -1;

// Run MPX test code.
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
