//===-- cpuid.cpp ---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

unsigned OverrideEAX = 0;
unsigned OverrideEBX = 0;
unsigned OverrideECX = 0;
unsigned OverrideEDX = 0;

int __cpu_indicator_init(void);

extern struct __processor_model {
  unsigned int __cpu_vendor;
  unsigned int __cpu_type;
  unsigned int __cpu_subtype;
  unsigned int __cpu_features[1];
} __cpu_model;

void OverrideCPUID(unsigned int EAX, unsigned int EBX, unsigned int ECX,
                   unsigned int EDX) {
  OverrideEAX = EAX;
  OverrideEBX = EBX;
  OverrideECX = ECX;
  OverrideEDX = EDX;

  __cpu_model.__cpu_vendor = 0;
  __cpu_indicator_init();
}

int __get_cpuid(unsigned int leaf, unsigned int *__eax, unsigned int *__ebx,
                unsigned int *__ecx, unsigned int *__edx) {
  *__eax = OverrideEAX;
  *__ebx = OverrideEBX;
  *__ecx = OverrideECX;
  *__edx = OverrideEDX;

  return 1;
}

int __get_cpuid_count(unsigned int __leaf, unsigned int __subleaf,
                      unsigned int *__eax, unsigned int *__ebx,
                      unsigned int *__ecx, unsigned int *__edx) {
  *__eax = OverrideEAX;
  *__ebx = OverrideEBX;
  *__ecx = OverrideECX;
  *__edx = OverrideEDX;

  return 1;
}
