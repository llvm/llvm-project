/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// #include <unistd.h>

// const char service_interp[] __attribute__((section(".interp"))) = "/lib/ld-linux.so.2";

#if defined(TARGET_LINUX_X8664) || defined(TARGET_OSX_X8664) || defined(TARGET_WIN_X8664)
#include "cpuid8664.h"
#endif

#ifndef	CPUIDX8664
#define CPUIDX8664(a)  __cpuid_##a
#endif // #ifndef CPUIDX8664

static	int CPUIDX8664(is_avx512)();
static	int CPUIDX8664(is_avx512vl)();
static	int CPUIDX8664(is_avx512f)();
static	int CPUIDX8664(is_avx2)();
static	int CPUIDX8664(is_avx)();
static	int CPUIDX8664(is_intel)();
static	int CPUIDX8664(is_amd)();
static	int CPUIDX8664(is_fma4)();
static	int CPUIDX8664(is_sse4a)();
static	int CPUIDX8664(is_sse41)();

extern const char *get_arch();

const char *get_arch() {
  char *ret = (char *) malloc(sizeof(char) * 10);
  strcpy(ret, "unknown");

  if (CPUIDX8664(is_avx512vl)() == 1) {
    strcpy(ret, "avx512");
  } else if (CPUIDX8664(is_avx512f)() == 1) {
    strcpy(ret, "avx512knl");
  } else if (CPUIDX8664(is_avx2)() == 1) {
    strcpy(ret, "avx2");
  } else if (CPUIDX8664(is_avx)() == 1) {
    if (CPUIDX8664(is_intel)() == 1) {
      strcpy(ret, "avx");
    }
    if (CPUIDX8664(is_amd)() == 1) {
      if (CPUIDX8664(is_fma4)() == 1) {
        strcpy(ret, "avxfma4");
      } else {
        strcpy(ret, "sse4");
      }
    }
  } else {
    if ((CPUIDX8664(is_sse4a)() == 1) || (CPUIDX8664(is_sse41)() == 1)) {
      strcpy(ret, "sse4");
    } else {
      strcpy(ret, "em64t");
    }
  }

  return ret;
}

// void lib_entry(int argc, char **argv) {
//   printf("Instruction Set: %s: ", get_arch());

//   _exit(0);
// }
