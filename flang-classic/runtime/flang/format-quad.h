#ifndef FORMAT_QUAD_H_
#define FORMAT_QUAD_H_

/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/*
 *  Formats a 128-bit IEEE-754 binary128 value into the indicated
 *  field using Fortran Fw.d, Ew.d, Dw.d, Ew.dEe, Dw.dEe, Gw.d,
 *  and Gw.dEe edit descriptors.  Always writes 'width' bytes.
 */
void __fortio_format_quad(char *out, int width,
                          const struct formatting_control *control,
                          float128_t x);

#endif /* FORMAT_QUAD_H_ */

