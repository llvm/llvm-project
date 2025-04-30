/* Offsets for data table for function pow.
   Copyright (C) 2014-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#ifndef D_POW_DATA_H
#define D_POW_DATA_H

#define _hsw_log2_table               	0
#define _hsw_dTe                      	8256
#define _hsw_dMantMask                	10304
#define _hsw_dOne                     	10368
#define _hsw_dCvtMask                 	10432
#define _hsw_dMinNorm                 	10496
#define _hsw_dMaxNorm                 	10560
#define _hsw_lRndBit                  	10624
#define _hsw_lRndMask                 	10688
#define _hsw_dc6                      	10752
#define _hsw_dc5                      	10816
#define _hsw_dc4                      	10880
#define _hsw_dc3                      	10944
#define _hsw_dc1                      	11008
#define _hsw_dc1h                     	11072
#define _hsw_dc2                      	11136
#define _hsw_dAbsMask                 	11200
#define _hsw_dDomainRange             	11264
#define _hsw_dShifter                 	11328
#define _hsw_dIndexMask               	11392
#define _hsw_dce4                     	11456
#define _hsw_dce3                     	11520
#define _hsw_dce2                     	11584
#define _hsw_dce1                     	11648
#define _rcp_t1                       	11712
#define _log2_t1                      	19968
#define _exp2_tbl                     	36416
#define _clv_1                        	38464
#define _clv_2                        	38528
#define _clv_3                        	38592
#define _clv_4                        	38656
#define _clv_5                        	38720
#define _clv_6                        	38784
#define _clv_7                        	38848
#define _cev_1                        	38912
#define _cev_2                        	38976
#define _cev_3                        	39040
#define _cev_4                        	39104
#define _cev_5                        	39168
#define _iMantissaMask                	39232
#define _i3fe7fe0000000000            	39296
#define _dbOne                        	39360
#define _iffffffff00000000            	39424
#define _db2p20_2p19                  	39488
#define _iHighMask                    	39552
#define _LHN                          	39616
#define _ifff0000000000000            	39680
#define _db2p45_2p44                  	39744
#define _NEG_INF                      	39808
#define _NEG_ZERO                     	39872
#define _d2pow52                      	39936
#define _d1div2pow111                 	40000
#define _HIDELTA                      	40064
#define _LORANGE                      	40128
#define _ABSMASK                      	40192
#define _INF                          	40256
#define _DOMAINRANGE                  	40320
#define _iIndexMask                   	40384
#define _iIndexAdd                    	40448
#define _i3fe7fe00                    	40512
#define _i2p20_2p19                   	40576
#define _iOne                         	40640
#define _jIndexMask                   	40704

.macro double_vector offset value
/* clang integrated assembler doesn't think subtract yields an absolute, skip.  */
#if !defined(__clang__)
.if .-__svml_dpow_data != \offset
.err
.endif
#endif
.rept 8
.quad \value
.endr
.endm

.macro float_vector offset value
/* clang integrated assembler doesn't think subtract yields an absolute, skip.  */
#if !defined(__clang__)
.if .-__svml_dpow_data != \offset
.err
.endif
#endif
.rept 16
.long \value
.endr
.endm

#endif
