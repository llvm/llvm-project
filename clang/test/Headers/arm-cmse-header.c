// RUN: %clang_cc1 -triple thumbv8m.base-eabi  -fsyntax-only -ffreestanding        %s -verify -mcmse
// RUN: %clang_cc1 -triple thumbv8m.base-eabi  -fsyntax-only -ffreestanding -x c++ %s -verify -mcmse
// expected-no-diagnostics

#include "system_reserved_names.h"

#include <arm_cmse.h>

typedef void (*callback_t)(void);

void func(callback_t fptr, void *p)
{
  cmse_TT(p);
  cmse_TTT(p);
  cmse_TTA(p);
  cmse_TTAT(p);

  cmse_TT_fptr(fptr);
  cmse_TTT_fptr(fptr);
  cmse_TTA_fptr(fptr);
  cmse_TTAT_fptr(fptr);
}
