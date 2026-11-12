// RUN: not %clang_cc1 -triple thumbv8m.base-eabi -fsyntax-only %s 2>&1 | FileCheck --check-prefix=CHECK-c %s
// RUN: not %clang_cc1 -triple thumbv8m.base-eabi -fsyntax-only -x c++ %s 2>&1 | FileCheck --check-prefix=CHECK-cpp %s

#include <arm_cmse.h>

typedef void (*callback_t)(void);

void func(callback_t fptr, void *p)
{
  cmse_TT(p);
  cmse_TTT(p);
  cmse_TT_fptr(fptr);
  cmse_TTT_fptr(fptr);

  cmse_TTA(p);
  cmse_TTAT(p);
  cmse_TTA_fptr(fptr);
  cmse_TTAT_fptr(fptr);
// CHECK-c: error: call to undeclared function 'cmse_TTA'
// CHECK-c: error: call to undeclared function 'cmse_TTAT'
// CHECK-c: error: call to undeclared function 'cmse_TTA_fptr'
// CHECK-c: error: call to undeclared function 'cmse_TTAT_fptr'
// CHECK-cpp: error: use of undeclared identifier 'cmse_TTA'
// CHECK-cpp: error: use of undeclared identifier 'cmse_TTAT'
// CHECK-cpp: error: use of undeclared identifier 'cmse_TTA_fptr'
// CHECK-cpp: error: use of undeclared identifier 'cmse_TTAT_fptr'
}
