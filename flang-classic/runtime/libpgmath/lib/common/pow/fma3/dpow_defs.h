/* 
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */


  
#define SGN_MASK_D                0x8000000000000000

#define INT2DBL_HI_D              0x4330000000000000
#define INT2DBL_LO_D              0x0000000080000000
#define INT2DBL_D                 0x4330000080000000


#define UPPERBOUND_1_D            0x4086232B00000000
#define UPPERBOUND_2_D            0x4087480000000000
#define ZERO_D                    0.0
#define INF_D                     0x7FF0000000000000
#define HI_ABS_MASK_D             0x7FFFFFFF00000000
#define ABS_MASK_D                0x7FFFFFFFFFFFFFFF
#define MULT_CONST_D              0x3FF0000000000000

#define D52_D                     52

#define HI_MASK_D                 0xFFFFFFFF00000000
#define SGN_EXP_MASK_D            0x7FF0000000000000
#define NEG_ONE_D                 -1.0
#define HALF_D                    0.5
#define SGN_MASK_D                0x8000000000000000
#define INF_FAKE_D                0x7FF00000FFFFFFFF
#define NAN_MASK_D                0xFFF8000000000000
#define NEG_ONE_CONST_D           0x3FF0000000000000

#define TEN_23_D                  1023
#define ELEVEN_D                  11

/*  log constants   */
#define LOG_POLY_6_D              6.6253631649203309E-2/65536.0
#define LOG_POLY_5_D              6.6250935587260612E-2/16384.0
#define LOG_POLY_4_D              7.6935437806732829E-2/4096.0
#define LOG_POLY_3_D              9.0908878711093280E-2/1024.0
#define LOG_POLY_2_D              1.1111111322892790E-1/256.0
#define LOG_POLY_1_D              1.4285714284546502E-1/64.0
#define LOG_POLY_0_D              2.0000000000003113E-1/16.0

#define CC_CONST_Y_D              3.3333333333333331E-1/4.0
#define CC_CONST_X_D              -9.8201492846582465E-18/4.0

#define EXPO_MASK_D               0xFFF0000000000000
#define HI_CONST_1_D              0x800FFFFFFFFFFFFF
#define HI_CONST_2_D              0x3FF0000000000000
#define HALFIFIER_D               0x0010000000000000
#define HI_THRESH_D               0x3FF6A09E00000000

#define ONE_F_D                   1.0
#define TWO_D                     2.0

#define LN2_HI_D                  6.9314718055994529e-1
#define LN2_LO_D                  2.3190468138462996e-17
/*  log constants   */


/*  exp constants   */
#define L2E_D                      1.4426950408889634e+0
#define NEG_LN2_HI_D               -6.9314718055994529e-1
#define NEG_LN2_LO_D               -2.3190468138462996e-17

#define EXP_POLY_B_D               2.5022322536502990E-008
#define EXP_POLY_A_D               2.7630903488173108E-007
#define EXP_POLY_9_D               2.7557514545882439E-006
#define EXP_POLY_8_D               2.4801491039099165E-005
#define EXP_POLY_7_D               1.9841269589115497E-004
#define EXP_POLY_6_D               1.3888888945916380E-003
#define EXP_POLY_5_D               8.3333333334550432E-003
#define EXP_POLY_4_D               4.1666666666519754E-002
#define EXP_POLY_3_D               1.6666666666666477E-001
#define EXP_POLY_2_D               5.0000000000000122E-001
#define EXP_POLY_1_D               1.0000000000000000E+000
#define EXP_POLY_0_D               1.0000000000000000E+000

#define DBL2INT_CVT_D              6755399441055744.0
/*  exp constants   */


/*  pow constants   */
#define ONE_D                      1

#define ALL_ONES_EXPONENT_D        0x7FF0000000000000
/*  pow constants   */


