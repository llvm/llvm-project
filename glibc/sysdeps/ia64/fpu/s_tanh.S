.file "tanh.s"


// Copyright (c) 2001 - 2005, Intel Corporation
// All rights reserved.
//
// Contributed 2001 by the Intel Numerics Group, Intel Corporation
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// * The name of Intel Corporation may not be used to endorse or promote
// products derived from this software without specific prior written
// permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL INTEL OR ITS
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Intel Corporation is the author of this code, and requests that all
// problem reports or change requests be submitted to it directly at
// http://www.intel.com/software/products/opensource/libraries/num.htm.
//
// History
//==============================================================================
// 05/30/01  Initial version
// 12/04/01  Rewritten version with erf-like algorithm.
//           Performance improved.
// 05/20/02  Cleaned up namespace and sf0 syntax
// 08/14/02  Changed mli templates to mlx
// 02/10/03  Reordered header: .section, .global, .proc, .align
// 03/31/05  Reformatted delimiters between data tables
//
// API
//==============================================================================
// double tanh(double)
//
// Overview of operation
//==============================================================================
//
// Algorithm description
// ---------------------
//
// There are 4 paths:
//
// 1. Special path: x = 0, Inf, NaNs, denormals
//    Return tanh(x) = +/-0.0 for zeros
//    Return tanh(x) = QNaN for NaNs
//    Return tanh(x) = sign(x)*1.0 for Inf
//    Return tanh(x) = x + x^2   for - denormals
//    Return tanh(x) = x - x^2   for + denormals
//
// 2. Near zero path: 0.0 < |x| < 0.25
//    Return tanh(x) = x + x^3*A3 + ... + x^19*A19
//
// 3. Main path: 0.25 <= |x| < 19.0625
//    For several ranges of 0.25 <= |x| < 19.0625
//    Return tanh(x) = sign(x)*(A0 + y*A1 + y^2*A2 +
//                                       + y^3*A3 + ... + y^19*A19)
//    where y = (|x|/a) - b
//
//    For each range there is particular set of coefficients.
//    Below is the list of ranges:
//    1/4  <= |x| < 1/2     a = 0.25, b = 1.0
//    1/2  <= |x| < 1.0     a = 0.5,  b = 1.0
//    1.0  <= |x| < 2.0     a = 1.0,  b = 1.0
//    2.0  <= |x| < 3.25    a = 2.0,  b = 1.0
//    3.25 <= |x| < 4.0     a = 2.0,  b = 2.0
//    4.0  <= |x| < 6.5     a = 4.0,  b = 1.0
//    6.5  <= |x| < 8.0     a = 4.0,  b = 2.0
//    8.0  <= |x| < 13.0    a = 8.0,  b = 1.0
//    13.0 <= |x| < 16.0    a = 8.0,  b = 2.0
//    16.0 <= |x| < 19.0625 a = 16.0, b = 1.0
//    ( [3.25;4.0], [6.5;8.0], [13.0;16.0] subranges separated
//                               for monotonicity issues resolve )
//
// 4. Saturation path: 19.0625 <= |x| < +INF
//    Return tanh(x) = sign(x)*(1.0 - tiny_value)
//    (tiny_value ~ 2^(-63))
//
// Registers used
//==============================================================================
// Floating Point registers used:
// f8 = input, output
// f32 -> f64
//
// General registers used:
// r32 -> r51, r2, r3
//
// Predicate registers used:
// p6, p8, p10, p11, p12, p14, p15
// p6           arg is zero, denormal or special IEEE
// p8           to filter out case when signd(x) > 1.625
// p10          to filter out case when |x| < 0.25
// p11          to filter out case when signd(x) <= 1.625
// p12          to filter out case when |x| >= 19.0625
// p14          set to 1 for positive x
// p15          set to 1 for negative x

// Assembly macros
//==============================================================================
rDataPtr           = r2
rDataPtr1          = r3

rBias              = r33
rCoeffAddr3        = r34
rThreeAndQ         = r35
rCoeffAddr2        = r36
rMask              = r37
rArg               = r38
rSignBit           = r39
rAbsArg            = r40
rSaturation        = r41
rIndex             = r42
rCoeffAddr1        = r43
rCoeffAddr4        = r44
rShiftedArg        = r45
rShiftedArgMasked  = r46
rBiasedExpOf4      = r47
rShiftedAbsArg     = r48
rArgSgnd           = r49
r1625Sgnd          = r50
rTwo               = r51

//==============================================================================
fA0                = f32
fA1                = f33
fA2                = f34
fA3                = f35
fA4                = f36
fA5                = f37
fA6                = f38
fA7                = f39
fA8                = f40
fA9                = f41
fA10               = f42
fA11               = f43
fA12               = f44
fA13               = f45
fA14               = f46
fA15               = f47
fA16               = f48
fA17               = f49
fA18               = f50
fA19               = f51
fArgSqr            = f52
fArgAbsNorm        = f53
fSignumX           = f54
fRes               = f55
fThreeAndQ         = f56
fArgAbs            = f57
fTSqr              = f58
fTQuadr            = f59
fTDeg3             = f60
fTDeg7             = f61
fArgAbsNormSgn     = f62
fTQuadrSgn         = f63
fTwo               = f64

// Data tables
//==============================================================================
RODATA

.align 16

LOCAL_OBJECT_START(tanh_data)
// CAUTION: The order of these table coefficients shouldn't be changed!

// Main path coefficients:
// Coefficients ##0..15 ("main" coefficient tables)
// Polynomial coefficients for the tanh(x), 0.25 <= |x| < 0.5
data8 0xE9D218BC9A3FB55A, 0x00003FC7 //A19
data8 0xC8C0D38687F36EBA, 0x00003FCE //A18
data8 0xA2663E519FAC8A43, 0x0000BFD2 //A17
data8 0xD913F0490674B0DF, 0x00003FD3 //A16
data8 0xF75D84789DE0AE52, 0x00003FD6 //A15
data8 0xACB3C40EEF3A06F0, 0x0000BFD9 //A14
data8 0xEBD7F5DC02CFD5BA, 0x0000BFDB //A13
data8 0x8B52CDF66D709E2A, 0x00003FDF //A12
data8 0x9EC21F28E05C4A3E, 0x00003FE0 //A11
data8 0xC412B44D0176F3ED, 0x0000BFE4 //A10
data8 0x97BF35A34DD1EA4C, 0x0000BFE0 //A9
data8 0xF89F5B39E3A3AA36, 0x00003FE9 //A8
data8 0xF2BA654BCEEBA433, 0x0000BFEA //A7
data8 0x8E1C15876AA589AD, 0x0000BFEF //A6
data8 0x942226246A8C2A86, 0x00003FF1 //A5
data8 0x8F06D9FF7DB47261, 0x00003FF4 //A4
//
// Polynomial coefficients for the tanh(x), 0.5 <= |x| < 1.0
data8 0xC4A7B8FB672A8520, 0x00003FDC //A19
data8 0xA20724B847E13499, 0x0000BFE0 //A18
data8 0xE17DB53F02E4D340, 0x00003FE2 //A17
data8 0x90264A1012F4CA6F, 0x0000BFE4 //A16
data8 0xEBEC9F776F0BF415, 0x0000BFE0 //A15
data8 0x89AF912B305B45A4, 0x00003FE7 //A14
data8 0xB4A960B81F5EC36A, 0x0000BFE7 //A13
data8 0x969A4E95B2DA86B5, 0x0000BFEA //A12
data8 0x8A3FC0EC082305CB, 0x00003FEC //A11
data8 0x83D7795BCBE24373, 0x00003FEC //A10
data8 0xDCBF42AEB82932EC, 0x0000BFEF //A9
data8 0x83318E61ECAFD804, 0x00003FF0 //A8
data8 0xEA4DE5746975A914, 0x00003FF2 //A7
data8 0xCE63E8FA6B96480B, 0x0000BFF4 //A6
data8 0xDF017BE0D4FE45D8, 0x0000BFF4 //A5
data8 0xA8A0C6E2226DF3CD, 0x00003FF8 //A4
//
// Polynomial coefficients for the tanh(x), 1.0 <= |x| < 2.0
data8 0x8E89D2EBFDAA160B, 0x00003FE9 //A19
data8 0xDD9226310A272046, 0x0000BFEC //A18
data8 0xA038042D28B0D665, 0x00003FEF //A17
data8 0x8C04796F03516306, 0x0000BFF1 //A16
data8 0x9CD6A9CB4E90A2FD, 0x00003FF2 //A15
data8 0xC8980E166F5A84FD, 0x0000BFF2 //A14
data8 0x9ADFE65F56B7BCFD, 0x00003FED //A13
data8 0x8B11FDFB5D0A7B96, 0x00003FF4 //A12
data8 0x8209A125E829CBFA, 0x0000BFF5 //A11
data8 0xCF38AAC17B85BD76, 0x00003FF1 //A10
data8 0xD5C2E248D8AB99AB, 0x00003FF6 //A9
data8 0xE12BE2785727F2D6, 0x0000BFF7 //A8
data8 0x9FC9EF90F87BF1E2, 0x00003FF6 //A7
data8 0x9B02FE0DAF42C08F, 0x00003FF9 //A6
data8 0xBDACE06F531D9491, 0x0000BFFA //A5
data8 0xE3048AD1DB2F648C, 0x00003FF9 //A4
//
// Polynomial coefficients for the tanh(x), 2.0 <= |x| < 3.25
data8 0x856EC3B0330A385A, 0x00003FEB //A19
data8 0xC641D69DAE2D429C, 0x0000BFF2 //A18
data8 0xC683EB0BE1343FFF, 0x00003FF5 //A17
data8 0xC358954224E4E823, 0x0000BFF7 //A16
data8 0xF813A8D6D396BC5F, 0x00003FF8 //A15
data8 0xE0ECDFED078D37D6, 0x0000BFF9 //A14
data8 0x950E4E619855E316, 0x00003FFA //A13
data8 0x8453B8F93370FB58, 0x0000BFFA //A12
data8 0xFDBA28430AEC95BA, 0x00003FF7 //A11
data8 0x9371AAC1FDB1E664, 0x00003FFA //A10
data8 0xAC972DA97782D88A, 0x0000BFFB //A9
data8 0xE18F47B10B9CE1BC, 0x00003FFB //A8
data8 0xAB7C81230BF13BC6, 0x0000BFFB //A7
data8 0xA6CAAD4A3E31A7D5, 0x0000BFF8 //A6
data8 0x9CABD76D1D5C3878, 0x00003FFC //A5
data8 0x92906D077941CAA9, 0x0000BFFD //A4
//
// Polynomial coefficients for the tanh(x), 4.0 <= |x| < 6.5
data8 0x9232D19F71709AC9, 0x0000BFF5 //A19
data8 0x819E31323F5DD3F8, 0x00003FF8 //A18
data8 0xDA8E1CDB8D23DC29, 0x0000BFF9 //A17
data8 0xE97C7CD8FC0486D8, 0x00003FFA //A16
data8 0xB0C4AD234D88C9F2, 0x0000BFFB //A15
data8 0xC5989BFB28FDE267, 0x00003FFB //A14
data8 0x9B26520EC4EFEE8E, 0x0000BFFB //A13
data8 0xC4B6F758AD21E574, 0x00003FF9 //A12
data8 0xCC36E3FFA10D2CFF, 0x00003FFA //A11
data8 0x8738696FB06A5CED, 0x0000BFFC //A10
data8 0xD31981825BF39228, 0x00003FFC //A9
data8 0x82C58FB9BEE43992, 0x0000BFFD //A8
data8 0x88D5AAE49164B6F3, 0x00003FFD //A7
data8 0xF4CA0B968AF2DDE2, 0x0000BFFC //A6
data8 0xB99874B482BD17EE, 0x00003FFC //A5
data8 0xE93FB2F99431DC1D, 0x0000BFFB //A4
//
// Polynomial coefficients for the tanh(x), 8.0 <= |x| < 13.0
data8 0xAAA9EB7EADA85CEC, 0x00003FF5 //A19
data8 0x980C80EE05A6BE78, 0x0000BFF8 //A18
data8 0x818DA9F5396390A5, 0x00003FFA //A17
data8 0x8D8CC21E23D8A6A2, 0x0000BFFB //A16
data8 0xE0EC19E55A886765, 0x00003FFB //A15
data8 0x8C11197A7E6244C5, 0x0000BFFC //A14
data8 0x901D2BF203C2F7F3, 0x00003FFC //A13
data8 0xFEACAEE66EE803E5, 0x0000BFFB //A12
data8 0xC684E4925E318C3F, 0x00003FFB //A11
data8 0x8A9D8A970565F28D, 0x0000BFFB //A10
data8 0xAE34C61DE5CEA4D4, 0x00003FFA //A9
data8 0xC44C5714BD6208A0, 0x0000BFF9 //A8
data8 0xC4612F7D6C8BDB79, 0x00003FF8 //A7
data8 0xABD91DCE40D5EECB, 0x0000BFF7 //A6
data8 0x80E375C1B847B72F, 0x00003FF6 //A5
data8 0xA11C7DD978CF700A, 0x0000BFF4 //A4
//
// Polynomial coefficients for the tanh(x), 16.0 <= |x| < 19.0625
data8 0xE29D17C510F86F6B, 0x00003FF3 //A19
data8 0x88FE52EB39A3A98C, 0x0000BFF5 //A18
data8 0xA406547E50360693, 0x00003FF5 //A17
data8 0x83E6260B71C6D7DE, 0x0000BFF5 //A16
data8 0xA36AB5B0CBC97B85, 0x00003FF4 //A15
data8 0xA94931E0B7BA6C14, 0x0000BFF3 //A14
data8 0x9A4596DAF350AD63, 0x00003FF2 //A13
data8 0xFE47643F375AECA5, 0x0000BFF0 //A12
data8 0xBF8433C5ABEE63B1, 0x00003FEF //A11
data8 0x83CEE05D7AE90A0A, 0x0000BFEE //A10
data8 0xA4CC45480BCEB02D, 0x00003FEC //A9
data8 0xB967CBDCBC16CB10, 0x0000BFEA //A8
data8 0xB9681B214EDC098D, 0x00003FE8 //A7
data8 0xA23B20D87B80DFA8, 0x0000BFE6 //A6
data8 0xF358B2C46F10CBAF, 0x00003FE3 //A5
data8 0x98176FD06229A385, 0x0000BFE1 //A4
//
// Binary subranges
// Polynomial coefficients for the tanh(x), 3.25 <= |x| < 4.0
data8 0xEF2EE841288F6706, 0x00003FE9 //A19
data8 0xE65D5B74B85F82A6, 0x00003FEB //A18
data8 0xE495FC21E42A79FF, 0x00003FEA //A17
data8 0xF99B267A913CF3E5, 0x00003FEC //A16
data8 0xFE3D700F4A0A0FDE, 0x0000BFEC //A15
data8 0x8F91BB4EE4E4EA52, 0x00003FEE //A14
data8 0xBCA9F41A5C6EF8BA, 0x0000BFEE //A13
data8 0xF93E00884027A9CF, 0x00003FED //A12
data8 0xC4D4036A61BABC2F, 0x00003FEF //A11
data8 0x86CC2AD1AD47C7D5, 0x0000BFF2 //A10
data8 0xD3065DEF4CE9AD32, 0x00003FF3 //A9
data8 0x82C44125F568D54E, 0x0000BFF5 //A8
data8 0x88D588729BAF14CA, 0x00003FF6 //A7
data8 0xF4CA0661307243C7, 0x0000BFF6 //A6
data8 0xB998746D57061F74, 0x00003FF7 //A5
data8 0xE93FB2F482327C19, 0x0000BFF7 //A4
//
// Polynomial coefficients for the tanh(x), 6.5 <= |x| < 8.0
data8 0xEB189B71ADC40BE2, 0x00003FEA //A19
data8 0xA60B46F9FF6DC2DF, 0x00003FEA //A18
data8 0xBB061CDD9F368B9D, 0x00003FEC //A17
data8 0x841E08BDF5429991, 0x0000BFEC //A16
data8 0xDD33990B433F25BE, 0x00003FED //A15
data8 0xBA5DE6B870F0A2BB, 0x0000BFEE //A14
data8 0xA71D489AAA6DACF0, 0x00003FEF //A13
data8 0x874CCB2B8F3FBC0E, 0x0000BFF0 //A12
data8 0xCB1D2E9754EA534A, 0x00003FF0 //A11
data8 0x8BA5ABB53BA6ABCF, 0x0000BFF1 //A10
data8 0xAE91FD1C2391A32B, 0x00003FF1 //A9
data8 0xC465A74B798E5761, 0x0000BFF1 //A8
data8 0xC4666152397D15C1, 0x00003FF1 //A7
data8 0xABD9E63CA575B950, 0x0000BFF1 //A6
data8 0x80E38B18E8D0F460, 0x00003FF1 //A5
data8 0xA11C80E20AAFDD3C, 0x0000BFF0 //A4
//
// Polynomial coefficients for the tanh(x), 13.0 <= |x| < 16.0
data8 0xBECD0AF7E22E5594, 0x00003FE9 //A19
data8 0xE2834E2D68C1128C, 0x00003FEA //A18
data8 0x97B117611B317379, 0x00003FEB //A17
data8 0xEE91A0D39A772F6B, 0x00003FEA //A16
data8 0x92F6EC377DCADA4F, 0x00003FEA //A15
data8 0xD8FCCD6A3277FAB7, 0x00003FE8 //A14
data8 0xC15AB9CB0C3DCFE0, 0x00003FE7 //A13
data8 0xC3C659704A7147CD, 0x00003FE2 //A12
data8 0xFA17F09D27C97912, 0x00003FE4 //A11
data8 0xF664147182B94788, 0x0000BFE3 //A10
data8 0xA6C89FA741464DA1, 0x00003FE3 //A9
data8 0xB90FE464A825EFA8, 0x0000BFE2 //A8
data8 0xB973AE0FD86EC024, 0x00003FE1 //A7
data8 0xA23A087F96846951, 0x0000BFE0 //A6
data8 0xF358D8A7FC012D5D, 0x00003FDE //A5
data8 0x98176E2309B7C73A, 0x0000BFDD //A4
//
// Coefficients ##16..19 ("tail" coefficient tables)
// Polynomial coefficients for the tanh(x), 0.25 <= |x| < 0.5
data8 0x838F209ABB9BA7B3, 0x0000BFF7 //A3
data8 0xEBC0AC78DA4FC500, 0x0000BFF8 //A2
data8 0xF0A4D02960B60E69, 0x00003FFC //A1
data8 0xFACBF534D0E42F8A, 0x00003FFC //A0
//
// Polynomial coefficients for the tanh(x), 0.5 <= |x| < 1.0
data8 0xC0ECBDC0A0D133A6, 0x0000BFF8 //A3
data8 0xBA13A076BF8E812F, 0x0000BFFB //A2
data8 0xC954A37D1A1CA070, 0x00003FFD //A1
data8 0xEC9A9EBAB4579B29, 0x00003FFD //A0
//
// Polynomial coefficients for the tanh(x), 1.0 <= |x| < 2.0
data8 0xD42E9175A6EA1397, 0x00003FFB //A3
data8 0xA3C361378A55CF56, 0x0000BFFD //A2
data8 0xD706E07CC8622983, 0x00003FFD //A1
data8 0xC2F7D5A8A79CA2AC, 0x00003FFE //A0
//
// Polynomial coefficients for the tanh(x), 2.0 <= |x| < 3.25
data8 0xAC7A7F8776817C7E, 0x00003FFD //A3
data8 0x8B7CE95E69FCFE9A, 0x0000BFFD //A2
data8 0x90B161317028D995, 0x00003FFC //A1
data8 0xF6CA82F0DE1E9E9A, 0x00003FFE //A0
//
// Polynomial coefficients for the tanh(x), 4.0 <= |x| < 6.5
data8 0xE9E072407BC22DC6, 0x00003FFA //A3
data8 0xAFA4A913D8E6BB4A, 0x0000BFF9 //A2
data8 0xAFC2D6A885BAA875, 0x00003FF7 //A1
data8 0xFFD40B84505A10B2, 0x00003FFE //A0
//
// Polynomial coefficients for the tanh(x), 8.0 <= |x| < 13.0
data8 0xA11C8A1FED168CD5, 0x00003FF2 //A3
data8 0xF1AAD6B02063A5F5, 0x0000BFEF //A2
data8 0xF1AADA46AD341C34, 0x00003FEC //A1
data8 0xFFFFFC39548FC34B, 0x00003FFE //A0
//
// Polynomial coefficients for the tanh(x), 16.0 <= |x| < 19.0625
data8 0x98176FD1F0950C16, 0x00003FDE //A3
data8 0xE42327BB09C8B2A5, 0x0000BFDA //A2
data8 0xE42327BB0B154F13, 0x00003FD6 //A1
data8 0xFFFFFFFFFFF8DEE7, 0x00003FFE //A0
//
// Binary subranges
// Polynomial coefficients for the tanh(x), 3.25 <= |x| < 4.0
data8 0xE9E072404329293B, 0x00003FF7 //A3
data8 0xAFA4A913D798300B, 0x0000BFF7 //A2
data8 0xAFC2D6A885B48567, 0x00003FF6 //A1
data8 0xFFD40B84505A10B4, 0x00003FFE //A0
//
// Polynomial coefficients for the tanh(x), 6.5 <= |x| < 8.0
data8 0xA11C8A63815F7A28, 0x00003FEF //A3
data8 0xF1AAD6B65B0EBF53, 0x0000BFED //A2
data8 0xF1AADA46E799831F, 0x00003FEB //A1
data8 0xFFFFFC39548FC348, 0x00003FFE //A0
//
// Polynomial coefficients for the tanh(x), 13.0 <= |x| < 16.0
data8 0x98176FE982140A59, 0x00003FDB //A3
data8 0xE42327B9B0D7202F, 0x0000BFD8 //A2
data8 0xE42327BB13076BD6, 0x00003FD5 //A1
data8 0xFFFFFFFFFFF8DEE7, 0x00003FFE //A0
//
// Polynomial coefficients for the tanh(x), 0.0 <= |x| < 0.25
// ('tanh_near_zero' path)
data8 0xBF2BA5D26E479D0C //A9
data8 0x3F4336D96F81EE26 //A8
data8 0xBF8226E34AE197B0 //A5
data8 0x3F9664F488148657 //A4
data8 0xAAAAAAAAAAAAAA99, 0x0000BFFD //A1
data8 0xBF57D91925BB5EE2 //A7
data8 0x3F6D6D36C3D5B7A1 //A6
data8 0xBFABA1BA1BA19D32 //A3
data8 0x3FC1111111111108 //A2
//
// 1.0 - 2^(-63)
// ('tanh_saturation' path)
data8 0xFFFFFFFFFFFFFFFF, 0x00003FFE
LOCAL_OBJECT_END(tanh_data)

// CAUTION: The order of table coefficients shouldn't be changed!


.section .text
GLOBAL_LIBM_ENTRY(tanh)
{ .mfi
      alloc          r32         = ar.pfs, 0, 20, 0, 0
      fmerge.se      fArgAbsNorm = f1, f8         // normalized x
      adds           rSignBit    = 0x1, r0        // Bit for sign removing
}
{ .mfi
      addl           rDataPtr    = @ltoff(tanh_data), gp // Data pointer
      fma.s1         fTwo        = f1, f1, f1            // 2.0 construct
      addl           rArgSgnd    = 0xfff, r0             // mask for exponent
};;

{ .mfi
      getf.d         rArg        = f8       // x in GR
      fclass.m       p6,p0       = f8, 0xEF // Filter 0, denormals and specials
                            // 0xEF = @qnan|@snan|@pos|@neg|@zero|@unorm|@inf
      shl            rArgSgnd    = rArgSgnd, 52  // mask for exponent
}
{ .mlx
      ld8            rDataPtr    = [rDataPtr]        // Real data pointer
      movl           r1625Sgnd   = 0xA000000000000   // 1.625 signd
      // 1.625 significand used to filter values greater than 3.25, 6.5, 13.0
      // to enter binary subranges
};;

{ .mfi
      addl           rBias       = 0x3FD00, r0       // bias of 0.25 << 8
      fma.s1         fArgSqr     = f8, f8, f0        // x^2
      shl            rSignBit    = rSignBit, 63      // mask for sign bit
}
{ .mlx
      addl           rMask       = 0x7FF00, r0          // Mask for index bits
      movl           rTwo        = 0x4000000000000000   // 2.0
};;

{ .mfi
      andcm          rArgSgnd    = rArg, rArgSgnd // Remove exponent
      nop.f          0
      shr.u          rShiftedArg = rArg, 44 // Select only necessary bits of arg
}
{ .mfb
      andcm          rAbsArg     = rArg, rSignBit     // Remove sign
      nop.f          0
(p6)  br.cond.spnt   _tanh_spec    // Branch to zero, denorm & specs
};;

{ .mfi
      and            rShiftedArgMasked = rShiftedArg, rMask // bias of x << 8
      fmerge.s       fArgAbs     = f1, f8                   // |x|
      shr            rShiftedAbsArg    = rAbsArg, 44 // Select only necessary
                                                     // bits of absolute arg
}
{ .mfi
      cmp.gt         p8, p11     = rArgSgnd, r1625Sgnd // p8 = 1 if
      // signd(x) > 1.625 - to filter values greater than 3.25, 6.5, 13.0
      nop.f          0
      nop.i          0
};;

{ .mfi
      sub            rIndex      = rShiftedArgMasked, rBias // index << 8
      nop.f          0
      cmp.lt         p10, p0     = rShiftedArgMasked, rBias // p10=1 if |x|<0.25
}
{ .mfb
(p8)  cmp.gt         p8, p11     = rAbsArg, rTwo // If arg is greater than 2.0?
                                       // (then we should use binary subranges)
      nop.f          0
(p10) br.cond.spnt   tanh_near_zero    // branch out if |x| < 0.25
};;

.pred.rel "mutex",p8,p11
{ .mfi
(p8)  add            rIndex      = 0x400, rIndex // Make pointer to binary
                                                 // subranges
(p11) fms.s1         fArgAbsNorm = fArgAbsNorm, f1, f1     // |x|/b - 1.0
      addl           rSaturation = 0x40331, r0 // shifted bits of 19.0625
}
{ .mfi
      nop.m          0
(p8)  fms.s1         fArgAbsNorm = fArgAbsNorm, f1, fTwo // |x|/b - 2.0
       // this is only for binary subranges [3.25;4], [6.5;8], [13.0;16]
      nop.i          0
}
;;

{ .mfi
      add            rCoeffAddr1 = rDataPtr, rIndex// coeff. ##0,2,..14
      nop.f          0
      nop.i          0
};;

{ .mfi
      adds           rCoeffAddr2 = 16, rCoeffAddr1 // Shifted pointer to coeffs
      fmerge.s       fSignumX    = f8, f1          // signum(x)
      nop.i          0
}
{ .mfb
      cmp.le         p12, p0     = rSaturation, rShiftedAbsArg // |x|>=19.0625?
      nop.f          0
(p12) br.cond.spnt   tanh_saturation          // branch out if x |x| >= 19.0625
};;

{.mfi
      ldfe           fA19        = [rCoeffAddr1], 32 // Load A19
      nop.f          0
      nop.i          0
}
{.mfi
      ldfe           fA18        = [rCoeffAddr2], 32 // Load A18
      nop.f          0
      adds           rCoeffAddr3 = 0xA00, rDataPtr   // Pointer to "tail"
                                                     // coefficients tables
};;

{.mfi
      ldfe           fA17        = [rCoeffAddr1], 32 // Load A17
      nop.f          0
      nop.i          0
}
{.mfi
      ldfe           fA16        = [rCoeffAddr2], 32 // Load A16
      nop.f          0
      nop.i          0
};;

{.mfi
      ldfe           fA15        = [rCoeffAddr1], 32 // Load A15
      fma.s1         fTSqr       = fArgAbsNorm, fArgAbsNorm, f0 // x^2
      shr.u          rIndex      = rIndex, 2 // Index for "tail" tables
}
{.mfi
      ldfe           fA14        = [rCoeffAddr2], 32 // Load A14
      nop.f          0
      adds           rCoeffAddr4 = 16, r0            // Shifter pointer
                                                     // to "tail" tables
};;

{.mfi
      ldfe           fA13        = [rCoeffAddr1], 32   // Load A13
      nop.f          0
      add            rCoeffAddr3 = rCoeffAddr3, rIndex // "tail" coeffs to load
                                                       // ##16..23
}
{.mfi
      ldfe           fA12        = [rCoeffAddr2], 32 // Load A12
      nop.f          0
      cmp.lt         p15, p14    = rArg, r0          // Arg positive (p14)
                                                     // or negative (p15)?
};;

{.mfi
      ldfe           fA11        = [rCoeffAddr1], 32        // Load A11
      nop.f          0
      add            rCoeffAddr4 = rCoeffAddr3, rCoeffAddr4 // shifted "tail"
                                                            // coeffs to load
}
{.mfi
      ldfe           fA10        = [rCoeffAddr2], 32 // Load A10
      nop.f          0
      nop.i          0
};;

{.mfi
      ldfe           fA9         = [rCoeffAddr1], 32 // Load A9
      nop.f          0
      nop.i          0
}
{.mfi
      ldfe           fA8         = [rCoeffAddr2], 32 // Load A8
      nop.f          0
      nop.i          0
};;

{.mfi
      ldfe           fA7         = [rCoeffAddr1], 32 // Load A7
      nop.f          0
      nop.i          0
}
{.mfi
      ldfe           fA6         = [rCoeffAddr2], 32 // Load A6
      nop.f          0
      nop.i          0
};;

{.mfi
      ldfe           fA5         = [rCoeffAddr1], 32 // Load A5
      fma.s1         fTDeg3      = fArgAbsNorm, fTSqr, f0 // x^3
      nop.i          0
}
{.mfi
      ldfe           fA4         = [rCoeffAddr2], 32 // Load A4
      fma.s1         fTQuadr     = fTSqr, fTSqr, f0  // x^4
      nop.i          0
};;

// Path #3 Polynomial Pol19(y) computation; y = fArgAbsNorm
{.mfi
      ldfe           fA3         = [rCoeffAddr3], 32            // Load A3
      fma.s1         fArgAbsNormSgn = fArgAbsNorm, fSignumX, f0 // sign(x)*x
      nop.i          0
}
{.mfi
      ldfe           fA2         = [rCoeffAddr4], 32            // Load A2
      nop.f          0
      nop.i          0
};;

{.mfi
      ldfe           fA1         = [rCoeffAddr3], 32       // Load A1
      fma.s1         fRes        = fA19, fArgAbsNorm, fA18 // Polynomial
      nop.i          0
}
{.mfi
      ldfe           fA0         = [rCoeffAddr4], 32       // Load A0
      nop.f          0
      nop.i          0
};;

{ .mfi
      nop.m          0
      fma.s1         fA17        = fA17, fArgAbsNorm, fA16  // Polynomial
      nop.i          0
};;

{ .mfi
      nop.m          0
      fma.s1         fA15        = fA15, fArgAbsNorm, fA14  // Polynomial
      nop.i          0
};;

{ .mfi
      nop.m          0
      fma.s1         fTDeg7      = fTDeg3, fTQuadr, f0     // Polynomial
      nop.i          0
}
{ .mfi
      nop.m          0
      fma.s1         fA13        = fA13, fArgAbsNorm, fA12 // Polynomial
      nop.i          0
};;

{ .mfi
      nop.m          0
      fma.s1         fA11        = fA11, fArgAbsNorm, fA10 // Polynomial
      nop.i          0
};;

{ .mfi
      nop.m          0
      fma.s1         fA9         = fA9, fArgAbsNorm, fA8   // Polynomial
      nop.i          0
};;

{ .mfi
      nop.m          0
      fma.s1         fRes        = fRes, fTSqr, fA17       // Polynomial
      nop.i          0
}
{ .mfi
      nop.m          0
      fma.s1         fA7         = fA7, fArgAbsNorm, fA6 // Polynomial
      nop.i          0
};;

{ .mfi
      nop.m          0
      fma.s1         fA5         = fA5, fArgAbsNorm, f0  // Polynomial
      nop.i          0
};;

{ .mfi
      nop.m          0
      fma.s1         fA15        = fA15, fTSqr, fA13     // Polynomial
      nop.i          0
}
{ .mfi
      nop.m          0
      fma.s1         fA4         = fA4, fArgAbsNorm, fA3 // Polynomial
      nop.i          0
};;

{ .mfi
      nop.m          0
      fma.s1         fA2         = fA2, fArgAbsNorm, fA1 // Polynomial
      nop.i          0
};;

{ .mfi
      nop.m          0
      fma.s1         fA11        = fA11, fTSqr, fA9 // Polynomial
      nop.i          0
};;

{ .mfi
      nop.m          0
      fma.s1         fA7         = fA7, fTSqr, fA5  // Polynomial
      nop.i          0
};;

{ .mfi
      nop.m          0
      fma.s1         fRes        = fRes, fTQuadr, fA15 // Polynomial
      nop.i          0
};;

{ .mfi
      nop.m          0
      fma.s1         fA4         = fA4, fTSqr, fA2     // Polynomial
      nop.i          0
};;

{ .mfi
      nop.m          0
      fma.s1         fRes        = fRes, fTQuadr, fA11 // Polynomial
      nop.i          0
};;

{ .mfi
      nop.m          0
      fma.s1         fA4         = fA7, fTDeg3, fA4    // Polynomial
      nop.i          0
};;

{ .mfi
      nop.m          0
      fma.s1         fRes        = fRes,  fTDeg7, fA4  // Polynomial
      nop.i          0
};;

{ .mfi
      nop.m          0
      // result for negative argument
(p15) fms.d.s0       f8          = fRes, fArgAbsNormSgn, fA0 // Polynomial
      nop.i          0
}
{ .mfb
      nop.m          0
      // result for positive argument
(p14) fma.d.s0       f8          = fRes, fArgAbsNormSgn, fA0 // Polynomial
      br.ret.sptk    b0
};;


// |x| < 0.25 Path /////////////////////////////////////////////////////////////
.align 32
tanh_near_zero:
{ .mfi
      adds           rCoeffAddr1 = 0xC80, rDataPtr      // address of A9
      fma.s0         fTSqr       = fArgSqr, fArgSqr, f0 // x^4
      nop.i          0
}
{ .mfi
      adds           rCoeffAddr2 = 0xCB0, rDataPtr      // address of A7
      nop.f          0
      nop.i          0
};;

{ .mfi
      ldfpd          fA9, fA8    = [rCoeffAddr1], 16 // Load A9, A8
      nop.f          0
      nop.i          0
}
{ .mfi
      ldfpd          fA7, fA6    = [rCoeffAddr2], 16 // Load A7, A6
      nop.f          0
      nop.i          0
};;

{ .mfi
      ldfpd          fA5, fA4    = [rCoeffAddr1], 16 // Load A5, A4
      nop.f          0
      nop.i          0
}
{ .mfi
      ldfpd          fA3, fA2    = [rCoeffAddr2], 16 // Load A3, A2
      nop.f          0
      nop.i          0
};;

{ .mfi
      ldfe           fA1         = [rCoeffAddr1] // Load A1
      nop.f          0
      nop.i          0
};;

{ .mfi
      nop.m          0
      fma.s1         fTQuadr     = fTSqr, fTSqr, f0 // x^4
      nop.i          0
};;

{ .mfi
      nop.m          0
      fma.s1         fRes        = fA9, fArgSqr, fA8 // Polynomial
      nop.i          0
}
{ .mfi
      nop.m          0
      fma.s1         fA7         = fA7, fArgSqr, fA6 // Polynomial
      nop.i          0
};;

{ .mfi
      nop.m          0
      fma.s1         fA3         = fA3, fArgSqr, fA2 // Polynomial
      nop.i          0
}
{ .mfi
      nop.m          0
      fma.s1         fA5         = fA5, fArgSqr, fA4 // Polynomial
      nop.i          0
};;

{ .mfi
      nop.m          0
      fma.s1         fA1         = fA1, fArgSqr, f0 // Polynomial
      nop.i          0
}
{ .mfi
      nop.m          0
      fma.s1         fTQuadrSgn  = fTQuadr, f8, f0  // x^4 * x
      nop.i          0
};;

{ .mfi
      nop.m          0
      fma.s1         fRes        = fRes, fTSqr, fA7 // Polynomial
      nop.i          0
};;

{ .mfi
      nop.m          0
      fma.s1         fA1         = fA3, fTSqr, fA1 // Polynomial
      nop.i          0
};;

{ .mfi
      nop.m          0
      fma.s1         fRes        = fRes, fTSqr, fA5 // Polynomial
      nop.i          0
};;

{ .mfi
      nop.m          0
      fma.s1         fRes        = fRes, fTQuadr, fA1 // Polynomial
      nop.i          0
};;

{ .mfb
      nop.m          0
      fma.d.s0       f8          = fRes, f8, f8 // x+x*Polynomial
      br.ret.sptk    b0                         // Exit for |x| < 0.25
};;





// 19.0625 <= |x| < +inf Saturation path ///////////////////////////////////////
.align 32
tanh_saturation:
{ .mfi
      adds           rDataPtr    = 0xCD0, rDataPtr  // address of A0
      nop.f          0
      nop.i          0
};;

{ .mfi
      ldfe           fA0         = [rDataPtr]       // Load  A0 = 2^(-63)
      nop.f          0
      nop.i          0
};;

{ .mfb
      nop.m          0
      fma.d.s0       f8          = fA0, fSignumX, f0 // sign(x)*(1.0-2^(-63))
      br.ret.sptk    b0                       // Exit for 19.0625 <=|x|< +inf
};;





//  0, denormals and special IEEE numbers path /////////////////////////////////
_tanh_spec:

{ .mfi
      cmp.lt         p15, p14    = rArg, r0 // Is arg negative (p15)
                                            // or positive p14)
      fclass.m       p6,p0       = f8, 0x23 // To filter infinities
                                          // 0x23 = @pos|@neg|@inf
      nop.i          0
};;

{ .mfi
      nop.m          0
      fclass.m       p7,p0       = f8, 0xC7 // To filter NaNs & Zeros
                                 // 0xC7 = @pos|@neg|@zero|@qnan|@snan
      nop.i          0
};;

{ .mfb
      nop.m          0
(p6)  fmerge.s       f8          = f8, f1     // +/-1 for INF args
(p6)  br.ret.spnt    b0                       // exit for x = INF
};;

{ .mfb
      nop.m          0
(p7)  fma.d.s0       f8          = f8, f1, f8    // +/-0 for 0 args
                                                 // and NaNs for NaNs
(p7)  br.ret.spnt    b0                          // exit for x = NaN or +/-0
};;

{ .mfi
      nop.m          0
      fnorm.s0       f8          = f8            // Normalize arg
      nop.i          0
};;

.pred.rel "mutex",p14,p15
{ .mfi
      nop.m          0
(p14) fnma.d.s0      f8          = f8, f8, f8  // res = r-r^2
      nop.i          0
}
{ .mfb
      nop.m          0
(p15) fma.d.s0       f8          = f8, f8, f8  // res = r+r^2
      br.ret.sptk    b0          // 0, denormals, specials return
};;

GLOBAL_LIBM_END(tanh)
libm_alias_double_other (tanh, tanh)
