#ifndef FORMAT_DOUBLE_H_
#define FORMAT_DOUBLE_H_

#include "float128.h"

/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/*
 *  These arbitrary integer values are equivalent to the code numbers
 *  used in the PGI Fortran run-time library so that its rounding
 *  mode flags can be passed through without translation.
 */
enum decimal_rounding {
  DECIMAL_ROUND_DEFAULT, /* same as COMPATIBLE */
  DECIMAL_ROUND_IN = 90, /* RZ: toward zero (truncation) */
  DECIMAL_ROUND_UP = 69, /* RU: toward +Inf */
  DECIMAL_ROUND_DOWN = 70, /* RD: toward -Inf */
  DECIMAL_ROUND_NEAREST = 71, /* RN: the usual IEEE "round to nearest", ties make even */
  DECIMAL_ROUND_COMPATIBLE = 72, /* RC: Fortran's round to nearest, ties diverge from zero */
  DECIMAL_ROUND_PROCESSOR_DEFINED = 73 /* RP: obey FPCR */
};

struct formatting_control {
  enum decimal_rounding rounding;
  int format_char; /* 'F', 'E' (including ES and EN), or 'D' */
  int fraction_digits; /* .d (for G, means 'significant digits') */
  int exponent_digits; /* Ee; 0 means E+nn or +nnn as needed */
  int scale_factor; /* k for kP */
  int plus_sign; /* '+' or NUL */
  int point_char; /* '.' or ',' */
  int ESN_format; /* 'S', 'N', or NUL for ES/EN variants of Ew.d */
  int no_minus_zero; /* '-' appears only if a nonzero digit does */
  int format_F0;     /* format is F0 */
  int format_G0;     /* format is G0 or G0.d */
};

/*
 *  Formats a 64-bit IEEE-754 double precision value into the
 *  indicated field using Fortran Fw.d, Ew.d, Dw.d, Ew.dEe, Dw.dEe,
 *  Gw.d, and Gw.dEe edit descriptors.  Always writes 'width' bytes.
 */
void __fortio_format_double(char *out, int width,
                            const struct formatting_control *control,
                            double x);

/*
 *  Formats a 128-bit IEEE-754 binary128 value into the indicated
 *  field using Fortran Fw.d, Ew.d, Dw.d, Ew.dEe, Dw.dEe, Gw.d,
 *  and Gw.dEe edit descriptors.  Always writes 'width' bytes.
 */
void __fortio_format_quad(char *out, int width,
                          const struct formatting_control *control,
                          float128_t x);

#endif /* FORMAT_DOUBLE_H_ */
