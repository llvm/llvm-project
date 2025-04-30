
/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef FE_RTLRTNSDESC_H_
#define FE_RTLRTNSDESC_H_

/**
 * \file  Runtime Library routine descriptions.
 *
 */
#include "gbldefs.h"

typedef struct {
  const char *baseNm;
  char fullNm[64];
  bool I8Descr;
  const char largeRetValPrefix[4];
} FtnRteRtn;

#endif /* FE_RTLRTNSDESC_H_ */
