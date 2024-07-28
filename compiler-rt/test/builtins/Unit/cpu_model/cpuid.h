//===-- cpuid.h -----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains functions that can be used to set a return value for
// the compiler-provided __get_cpuid and __get_cpuid_count functions so that
// specific values can be tested.
//
//===----------------------------------------------------------------------===//

void OverrideCPUID(unsigned int EAX, unsigned int EBX, unsigned int ECX,
                   unsigned int EDX);

int __get_cpuid(unsigned int leaf, unsigned int *__eax, unsigned int *__ebx,
                unsigned int *__ecx, unsigned int *__edx);

int __get_cpuid_count(unsigned int __leaf, unsigned int __subleaf,
                      unsigned int *__eax, unsigned int *__ebx,
                      unsigned int *__ecx, unsigned int *__edx);
