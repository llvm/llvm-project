// RUN: %clang_cc1 %s -fsyntax-only

#ifndef XVID_AUTO_INCLUDE

#define XVID_AUTO_INCLUDE
#define FUNC_H      H_Pass_16_C
#include "multiple-include.c"

#define FUNC_H      H_Pass_8_C

#include "multiple-include.c"
#undef XVID_AUTO_INCLUDE

typedef void ff();
typedef struct { ff *a;} S;

S s = { H_Pass_8_C };

#endif 

#if defined(XVID_AUTO_INCLUDE) && defined(REFERENCE_CODE)
#elif defined(XVID_AUTO_INCLUDE) && !defined(REFERENCE_CODE)

static void FUNC_H(){};
#undef FUNC_H

#endif
