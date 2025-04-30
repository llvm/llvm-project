.file "erfl.s"


// Copyright (c) 2001 - 2003, Intel Corporation
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
//==============================================================
// 11/21/01  Initial version
// 05/20/02  Cleaned up namespace and sf0 syntax
// 08/14/02  Changed mli templates to mlx
// 02/06/03  Reordered header: .section, .global, .proc, .align
//
// API
//==============================================================
// long double erfl(long double)
//
// Overview of operation
//==============================================================
//
// Algorithm description
// ---------------------
//
// There are 4 paths:
//
// 1. Special path: x = 0, Inf, NaNs, denormal
//    Return erfl(x) = +/-0.0 for zeros
//    Return erfl(x) = QNaN for NaNs
//    Return erfl(x) = sign(x)*1.0 for Inf
//    Return erfl(x) = (A0H+A0L)*x + x^2, ((A0H+A0L) = 2.0/sqrt(Pi))
//                                             for denormals
//
// 2. [0;1/8] path: 0.0 < |x| < 1/8
//    Return erfl(x) = x*(A1H+A1L) + x^3*A3 + ... + x^15*A15
//
// 3. Main path: 1/8 <= |x| < 6.53
//    For several ranges of 1/8 <= |x| < 6.53
//    Return erfl(x) = sign(x)*((A0H+A0L) + y*(A1H+A1L) + y^2*(A2H+A2L) +
//                                       + y^3*A3 + y^4*A4 + ... + y^25*A25 )
//    where y = (|x|/a) - b
//
//    For each range there is particular set of coefficients.
//    Below is the list of ranges:
//    1/8  <= |x| < 1/4     a = 0.125, b = 1.5
//    1/4  <= |x| < 1/2     a = 0.25,  b = 1.5
//    1/2  <= |x| < 1.0     a = 0.5,   b = 1.5
//    1.0  <= |x| < 2.0     a = 1.0,   b = 1.5
//    2.0  <= |x| < 3.25    a = 2.0,   b = 1.5
//    3.25 <= |x| < 4.0     a = 2.0,   b = 2.0
//    4.0  <= |x| < 6.53    a = 4.0,   b = 1.5
//    ( [3.25;4.0] subrange separated for monotonicity issues resolve )
//
// 4. Saturation path: 6.53 <= |x| < +INF
//    Return erfl(x) = sign(x)*(1.0 - tiny_value)
//    (tiny_value ~ 1e-1233)
//
// Implementation notes
// --------------------
//
// 1. Special path: x = 0, INF, NaNa, denormals
//
//    This branch is cut off by one fclass operation.
//    Then zeros+nans, infinities and denormals processed separately.
//    For denormals we had to use multiprecision A0 coefficient to reach
//    necessary accuracy: (A0H+A0L)*x-x^2
//
// 2. [0;1/8] path: 0.0 < |x| < 1/8
//
//    First coefficient of polynomial we must split to multiprecision too.
//    Also we can parallelise computations:
//    (x*(A1H+A1L)) calculated in parallel with "tail" (x^3*A3 + ... + x^15*A15)
//    Furthermore the second part is factorized using binary tree technique.
//
// 3. Main path: 1/8 <= |x| < 6.53
//
//    Multiprecision have to be performed only for first few
//    polynomial iterations (up to 3-rd x degree)
//    Here we use the same parallelisation way as above:
//    Split whole polynomial to first, "multiprecision" part, and second,
//    so called "tail", native precision part.
//
//    1) Multiprecision part:
//    [v1=(A0H+A0L)+y*(A1H+A1L)] + [v2=y^2*((A2H+A2L)+y*A3)]
//    v1 and v2 terms calculated in parallel
//
//    2) Tail part:
//    v3 = x^4 * ( A4 + x*A5 + ... + x^21*A25 )
//    v3 is splitted to 2 even parts (10 coefficient in each one).
//    These 2 parts are also factorized using binary tree technique.
//
//    So Multiprecision and Tail parts cost is almost the same
//    and we have both results ready before final summation.
//
// 4. Saturation path: 6.53 <= |x| < +INF
//
//    We use formula sign(x)*(1.0 - tiny_value) instead of simple sign(x)*1.0
//    just to meet IEEE requirements for different rounding modes in this case.
//
// Registers used
//==============================================================
// Floating Point registers used:
// f8 - input & output
// f32 -> f90

// General registers used:
// r2, r3, r32 -> r52

// Predicate registers used:
// p0, p6 -> p11, p14, p15

// p6  - arg is zero, denormal or special IEEE
// p7  - arg is in [4;8] binary interval
// p8  - arg is in [3.25;4] interval
// p9  - arg < 1/8
// p10 - arg is NOT in [3.25;4] interval
// p11 - arg in saturation domain
// p14 - arg is positive
// p15 - arg is negative

// Assembly macros
//==============================================================
rDataPtr           = r2
rTailDataPtr       = r3

rBias              = r33
rSignBit           = r34
rInterval          = r35

rArgExp            = r36
rArgSig            = r37
r3p25Offset        = r38
r2to4              = r39
r1p25              = r40
rOffset            = r41
r1p5               = r42
rSaturation        = r43
r3p25Sign          = r44
rTiny              = r45
rAddr1             = r46
rAddr2             = r47
rTailAddr1         = r48
rTailAddr2         = r49
rTailOffset        = r50
rTailAddOffset     = r51
rShiftedDataPtr    = r52

//==============================================================
fA0H               = f32
fA0L               = f33
fA1H               = f34
fA1L               = f35
fA2H               = f36
fA2L               = f37
fA3                = f38
fA4                = f39
fA5                = f40
fA6                = f41
fA7                = f42
fA8                = f43
fA9                = f44
fA10               = f45
fA11               = f46
fA12               = f47
fA13               = f48
fA14               = f49
fA15               = f50
fA16               = f51
fA17               = f52
fA18               = f53
fA19               = f54
fA20               = f55
fA21               = f56
fA22               = f57
fA23               = f58
fA24               = f59
fA25               = f60

fArgSqr            = f61
fArgCube           = f62
fArgFour           = f63
fArgEight          = f64

fArgAbsNorm        = f65
fArgAbsNorm2       = f66
fArgAbsNorm2L      = f67
fArgAbsNorm3       = f68
fArgAbsNorm4       = f69
fArgAbsNorm11      = f70

fRes               = f71
fResH              = f72
fResL              = f73
fRes1H             = f74
fRes1L             = f75
fRes1Hd            = f76
fRes2H             = f77
fRes2L             = f78
fRes3H             = f79
fRes3L             = f80
fRes4              = f81

fTT                = f82
fTH                = f83
fTL                = f84
fTT2               = f85
fTH2               = f86
fTL2               = f87

f1p5               = f88
f2p0               = f89
fTiny              = f90


// Data tables
//==============================================================
RODATA

.align 64
LOCAL_OBJECT_START(erfl_data)
////////// Main tables ///////////
_0p125_to_0p25_data: // exp = 2^-3
// Polynomial coefficients for the erf(x), 1/8 <= |x| < 1/4
data8 0xACD9ED470F0BB048, 0x0000BFF4 //A3 = -6.5937529303909561891162915809e-04
data8 0xBF6A254428DDB452 //A2H = -3.1915980570631852578089571182e-03
data8 0xBC131B3BE3AC5079 //A2L = -2.5893976889070198978842231134e-19
data8 0x3FC16E2D7093CD8C //A1H = 1.3617485043469590433318217038e-01
data8 0x3C6979A52F906B4C //A1L = 1.1048096806003284897639351952e-17
data8 0x3FCAC45E37FE2526 //A0H = 2.0911767705937583938791135552e-01
data8 0x3C648D48536C61E3 //A0L = 8.9129592834861155344147026365e-18
data8 0xD1FC135B4A30E746, 0x00003F90 //A25 = 6.3189963203954877364460345654e-34
data8 0xB1C79B06DD8C988C, 0x00003F97 //A24 = 6.8478253118093953461840838106e-32
data8 0xCC7AE121D1DEDA30, 0x0000BF9A //A23 = -6.3010264109146390803803408666e-31
data8 0x8927B8841D1E0CA8, 0x0000BFA1 //A22 = -5.4098171537601308358556861717e-29
data8 0xB4E84D6D0C8F3515, 0x00003FA4 //A21 = 5.7084320046554628404861183887e-28
data8 0xC190EAE69A67959A, 0x00003FAA //A20 = 3.9090359419467121266470910523e-26
data8 0x90122425D312F680, 0x0000BFAE //A19 = -4.6551806872355374409398000522e-25
data8 0xF8456C9C747138D6, 0x0000BFB3 //A18 = -2.5670639225386507569611436435e-23
data8 0xCDCAE0B3C6F65A3A, 0x00003FB7 //A17 = 3.4045511783329546779285646369e-22
data8 0x8F41909107C62DCC, 0x00003FBD //A16 = 1.5167830861896169812375771948e-20
data8 0x82F0FCB8A4B8C0A3, 0x0000BFC1 //A15 = -2.2182328575376704666050112195e-19
data8 0x92E992C58B7C3847, 0x0000BFC6 //A14 = -7.9641369349930600223371163611e-18
LOCAL_OBJECT_END(erfl_data)

LOCAL_OBJECT_START(_0p25_to_0p5_data)
// Polynomial coefficients for the erf(x), 1/4 <= |x| < 1/2
data8 0xF083628E8F7CE71D, 0x0000BFF6 //A3 = -3.6699405305266733332335619531e-03
data8 0xBF978749A434FE4E //A2H = -2.2977018973732214746075186440e-02
data8 0xBC30B3FAFBC21107 //A2L = -9.0547407100537663337591537643e-19
data8 0x3FCF5F0CDAF15313 //A1H = 2.4508820238647696654332719390e-01
data8 0x3C1DFF29F5AD8117 //A1L = 4.0653155218104625249413579084e-19
data8 0x3FD9DD0D2B721F38 //A0H = 4.0411690943482225790717166092e-01
data8 0x3C874C71FEF1759E //A0L = 4.0416653425001310671815863946e-17
data8 0xA621D99B8C12595E, 0x0000BFAB //A25 = -6.7100271986703749013021666304e-26
data8 0xBD7BBACB439992E5, 0x00003FAE //A24 = 6.1225362452814749024566661525e-25
data8 0xFF2FEFF03A98E410, 0x00003FB2 //A23 = 1.3192871864994282747963195183e-23
data8 0xAE8180957ABE6FD5, 0x0000BFB6 //A22 = -1.4434787102181180110707433640e-22
data8 0xAF0566617B453AA6, 0x0000BFBA //A21 = -2.3163848847252215762970075142e-21
data8 0x8F33D3616B9B8257, 0x00003FBE //A20 = 3.0324297082969526400202995913e-20
data8 0xD58AB73354438856, 0x00003FC1 //A19 = 3.6175397854863872232142412590e-19
data8 0xD214550E2F3210DF, 0x0000BFC5 //A18 = -5.6942141660091333278722310354e-18
data8 0xE2CA60C328F3BBF5, 0x0000BFC8 //A17 = -4.9177359011428870333915211291e-17
data8 0x88D9BB274F9B3873, 0x00003FCD //A16 = 9.4959118337089189766177270051e-16
data8 0xCA4A00AB538A2DB2, 0x00003FCF //A15 = 5.6146496538690657993449251855e-15
data8 0x9CC8FFFBDDCF9853, 0x0000BFD4 //A14 = -1.3925319209173383944263942226e-13
LOCAL_OBJECT_END(_0p25_to_0p5_data)

LOCAL_OBJECT_START(_0p5_to_1_data)
// Polynomial coefficients for the erf(x), 1/2 <= |x| < 1
data8 0xDB742C8FB372DBE0, 0x00003FF6 //A3 = 3.3485993187250381721535255963e-03
data8 0xBFBEDC5644353C26 //A2H = -1.2054957547410136142751468924e-01
data8 0xBC6D7215B023455F //A2L = -1.2770012232203569059818773287e-17
data8 0x3FD492E42D78D2C4 //A1H = 3.2146553459760363047337250464e-01
data8 0x3C83A163CAC22E05 //A1L = 3.4053365952542489137756724868e-17
data8 0x3FE6C1C9759D0E5F //A0H = 7.1115563365351508462453011816e-01
data8 0x3C8B1432F2CBC455 //A0L = 4.6974407716428899960674098333e-17
data8 0x95A6B92162813FF8, 0x00003FC3 //A25 = 1.0140763985766801318711038400e-18
data8 0xFE5EC3217F457B83, 0x0000BFC6 //A24 = -1.3789434273280972156856405853e-17
data8 0x9B49651031B5310B, 0x0000BFC8 //A23 = -3.3672435142472427475576375889e-17
data8 0xDBF73927E19B7C8D, 0x00003FCC //A22 = 7.6315938248752024965922341872e-16
data8 0xF55CBA3052730592, 0x00003FCB //A21 = 4.2563559623888750271176552350e-16
data8 0xA1DC9380DA82CFF6, 0x0000BFD2 //A20 = -3.5940500736023122607663701015e-14
data8 0xAAD1AE1067F3D577, 0x00003FD2 //A19 = 3.7929451192558641569555227613e-14
data8 0xCD1DB83F3B9D2090, 0x00003FD7 //A18 = 1.4574374961011929143375716362e-12
data8 0x87235ACB5E8BB298, 0x0000BFD9 //A17 = -3.8408559294899660346666452560e-12
data8 0xDA417B78FF9F46B4, 0x0000BFDC //A16 = -4.9625621225715971268115023451e-11
data8 0xF075762685484436, 0x00003FDE //A15 = 2.1869603559309150844390066920e-10
data8 0xB989FDB3795165C7, 0x00003FE1 //A14 = 1.3499740992928183247608593000e-09
LOCAL_OBJECT_END(_0p5_to_1_data)

LOCAL_OBJECT_START(_1_to_2_data)
// Polynomial coefficients for the erf(x), 1 <= |x| < 2.0
data8 0x8E15015F5B55BEAC, 0x00003FFC //A3 = 1.3875200409423426678618977531e-01
data8 0xBFC6D5A95D0A1B7E //A2H = -1.7839543383544403942764233761e-01
data8 0xBC7499F704C80E02 //A2L = -1.7868888188464394090788198634e-17
data8 0x3FBE723726B824A8 //A1H = 1.1893028922362935961842822508e-01
data8 0x3C6B77F399C2AD27 //A1L = 1.1912589318015368492508652194e-17
data8 0x3FEEEA5557137ADF //A0H = 9.6610514647531064991170524081e-01
data8 0x3C963D0DDD0A762F //A0L = 7.7155271023949055047261953350e-17
data8 0x8FAA405DAD409771, 0x0000BFDB //A25 = -1.6332824616946528652252813763e-11
data8 0x941386F4697976D8, 0x0000BFDD //A24 = -6.7337295147729213955410252613e-11
data8 0xBCBE75234530B404, 0x00003FDF //A23 = 3.4332329029092304943838374908e-10
data8 0xF55E2CE71A00D040, 0x00003FDF //A22 = 4.4632156034175937694868068394e-10
data8 0xA6CADFE489D2671F, 0x0000BFE3 //A21 = -4.8543000253822277507724949798e-09
data8 0xA4C69F11FEAFB3A8, 0x00003FE2 //A20 = 2.3978044150868471771557059958e-09
data8 0xD63441E3BED59703, 0x00003FE6 //A19 = 4.9873285553412397317802071288e-08
data8 0xDFDAED9D3089D732, 0x0000BFE7 //A18 = -1.0424069510877052249228047044e-07
data8 0xB47287FF165756A5, 0x0000BFE9 //A17 = -3.3610945128073834488448164164e-07
data8 0xCDAF2DC0A79A9059, 0x00003FEB //A16 = 1.5324673941628851136481785187e-06
data8 0x9FD6A7B2ECE8EDA9, 0x00003FEA //A15 = 5.9544479989469083598476592569e-07
data8 0xEC6E63BB4507B585, 0x0000BFEE //A14 = -1.4092398243085031882423746824e-05
LOCAL_OBJECT_END(_1_to_2_data)

LOCAL_OBJECT_START(_2_to_3p25_data)
// Polynomial coefficients for the erf(x), 2 <= |x| < 3.25
data8 0xCEDBA58E8EE6F055, 0x00003FF7 //A3 = 6.3128050215859026984338771121e-03
data8 0xBF5B60D5E974CBBD //A2H = -1.6710366233609740427984435840e-03
data8 0xBC0E11E2AEC18AF6 //A2L = -2.0376133202996259839305825162e-19
data8 0x3F32408E9BA3327E //A1H = 2.7850610389349567379974059733e-04
data8 0x3BE41010E4B3B224 //A1L = 3.3987633691879253781833531576e-20
data8 0x3FEFFFD1AC4135F9 //A0H = 9.9997790950300136092465663751e-01
data8 0x3C8EEAFA1E97EAE0 //A0L = 5.3633970564750967956196033852e-17
data8 0xBF9C6F2C6D7263C1, 0x00003FF0 //A25 = 4.5683639377039166585098497471e-05
data8 0xCB4167CC4798096D, 0x00003FF0 //A24 = 4.8459885139772945417160731273e-05
data8 0xE1394FECFE972D32, 0x0000BFF2 //A23 = -2.1479022581129892562916533804e-04
data8 0xC7F9E47581FC2A5F, 0x0000BFF2 //A22 = -1.9071211076537531370822343363e-04
data8 0xDD612EDFAA41BEAE, 0x00003FF2 //A21 = 2.1112405918671957390188348542e-04
data8 0x8C166AA4CB2AD8FD, 0x0000BFF4 //A20 = -5.3439165021555312536009227942e-04
data8 0xEFBE33D9F62B68D4, 0x0000BFF2 //A19 = -2.2863672131516067770956697877e-04
data8 0xCCB92F5D91562494, 0x00003FF5 //A18 = 1.5619154280865226092321881421e-03
data8 0x80A5DBE71D4BA0E2, 0x0000BFF6 //A17 = -1.9630109664962540123775799179e-03
data8 0xA0ADEB2D4C41347A, 0x0000BFF4 //A16 = -6.1294315248639348947483422457e-04
data8 0xB1F5D4911B911665, 0x00003FF7 //A15 = 5.4309165882071876864550213817e-03
data8 0xF2F3D8D21E8762E0, 0x0000BFF7 //A14 = -7.4143227286535936033409745884e-03
LOCAL_OBJECT_END(_2_to_3p25_data)

LOCAL_OBJECT_START(_4_to_6p53_data)
// Polynomial coefficients for the erf(x), 4 <= |x| < 6.53
data8 0xDF3151BE8652827E, 0x00003FD5 //A3 = 3.9646979666953349095427642209e-13
data8 0xBD1C4A9787DF888B //A2H = -2.5127788450714750484839908889e-14
data8 0xB99B35483E4603FD //A2L = -3.3536613901268985626466020210e-31
data8 0x3CD2DBF507F1A1F3 //A1H = 1.0468963266736687758710258897e-15
data8 0x398A97B60913B4BD //A1L = 1.6388968267515149775818013207e-31
data8 0x3FEFFFFFFFFFFFFF //A0H = 9.9999999999999988897769753748e-01
data8 0x3C99CC25E658129E //A0L = 8.9502895736398715695745861054e-17
data8 0xB367B21294713D39, 0x00003FFB //A25 = 8.7600127403270828432337605471e-02
data8 0xCEE3A423ADEC0F4C, 0x00003FFD //A24 = 4.0408051429309221404807497715e-01
data8 0xC389626CF2D727C0, 0x00003FFE //A23 = 7.6381507072332210580356159947e-01
data8 0xD15A03E082D0A307, 0x00003FFE //A22 = 8.1777977210259904277239787430e-01
data8 0x8FD3DA92675E8E00, 0x00003FFE //A21 = 5.6182638239203638864793584264e-01
data8 0xFD375E6EE167AA58, 0x00003FFC //A20 = 2.4728152801285544751731937424e-01
data8 0x89A9482FADE66AE1, 0x00003FFB //A19 = 6.7217410998398471333985773237e-02
data8 0xC62E1F02606C04DD, 0x00003FF7 //A18 = 6.0479785358923404401184993359e-03
data8 0xEE7BF2BE71CC531C, 0x0000BFF5 //A17 = -1.8194898432032114199803271708e-03
data8 0x8084081981CDC79C, 0x0000BFF5 //A16 = -9.8049734947701208487713246099e-04
data8 0x8975DFB834C118C3, 0x0000BFF0 //A15 = -3.2773123965143773578608926094e-05
data8 0x965DA4A80008B7BC, 0x0000BFEE //A14 = -8.9624997201558650125662820562e-06
LOCAL_OBJECT_END(_4_to_6p53_data)

LOCAL_OBJECT_START(_3p25_to_4_data)
// Polynomial coefficients for the erf(x), 3.25 <= |x| < 4
data8 0xB01D29846286CE08, 0x00003FEE //A3 = 1.0497207328743021499800978059e-05
data8 0xBEC10B1488AEB234 //A2H = -2.0317175474986489113480084279e-06
data8 0xBB7F19701B8B74F9 //A2L = -4.1159669348226960337518214996e-22
data8 0x3E910B1488AEB234 //A1H = 2.5396469343733111391850105348e-07
data8 0x3B4F1944906D5D60 //A1L = 5.1448487494628801547474934193e-23
data8 0x3FEFFFFFF7B91176 //A0H = 9.9999998458274208523732795584e-01
data8 0x3C70B2865615DB3F //A0L = 1.4482653192002495179309994964e-17
data8 0xA818D085D56F3021, 0x00003FEC //A25 = 2.5048394770210505593609705765e-06
data8 0xD9C5C509AAE5561F, 0x00003FEC //A24 = 3.2450636894654766492719395406e-06
data8 0x9682D71C549EEB07, 0x0000BFED //A23 = -4.4855801709974050650263470866e-06
data8 0xBC230E1EB6FBF8B9, 0x00003FEA //A22 = 7.0086469577174843181452303996e-07
data8 0xE1432649FF29D4DE, 0x0000BFEA //A21 = -8.3916747195472308725504497231e-07
data8 0xB40CEEBD2803D2F0, 0x0000BFEF //A20 = -2.1463694318102769992677291330e-05
data8 0xEAAB57ABFFA003EB, 0x00003FEF //A19 = 2.7974761309213643228699449426e-05
data8 0xFBFA4D0B893A5BFB, 0x0000BFEE //A18 = -1.5019043571612821858165073446e-05
data8 0xBB6AA248EED3E364, 0x0000BFF0 //A17 = -4.4683584873907316507141131797e-05
data8 0x86C1B3AE3E500ED9, 0x00003FF2 //A16 = 1.2851395412345761361068234880e-04
data8 0xB60729445F0C37B5, 0x0000BFF2 //A15 = -1.7359540313300841352152461287e-04
data8 0xCA389F9E707337B1, 0x00003FF1 //A14 = 9.6426575465763394281615740282e-05
LOCAL_OBJECT_END(_3p25_to_4_data)


//////// "Tail" tables //////////
LOCAL_OBJECT_START(_0p125_to_0p25_data_tail)
// Polynomial coefficients for the erf(x), 1/8 <= |x| < 1/4
data8 0x93086CBD21ED3962, 0x00003FCA //A13 = 1.2753071968462837024755878679e-16
data8 0x83CB5045A6D4B419, 0x00003FCF //A12 = 3.6580237062957773626379648530e-15
data8 0x8FCDB723209690EB, 0x0000BFD3 //A11 = -6.3861616307180801527566117146e-14
data8 0xCAA173F680B5D56B, 0x0000BFD7 //A10 = -1.4397775466324880354578008779e-12
data8 0xF0CEA934AD6AC013, 0x00003FDB //A9 = 2.7376616955640415767655526857e-11
data8 0x81C69F9D0B5AB8EE, 0x00003FE0 //A8 = 4.7212187567505249115688961488e-10
data8 0xA8B590298C20A194, 0x0000BFE4 //A7 = -9.8201697105565925460801441797e-09
data8 0x84F3DE72AC964615, 0x0000BFE8 //A6 = -1.2382176987480830706988411266e-07
data8 0xC01A1398868CC4BD, 0x00003FEC //A5 = 2.8625408039722670291121341583e-06
data8 0xCC43247F4410C54A, 0x00003FEF //A4 = 2.4349960762505993017186935493e-05
LOCAL_OBJECT_END(_0p125_to_0p25_data_tail)

LOCAL_OBJECT_START(_0p25_to_0p5_data_tail)
// Polynomial coefficients for the erf(x), 1/4 <= |x| < 1/2
data8 0x8CEAC59AF361B78A, 0x0000BFD6 //A13 = -5.0063802958258679384986669123e-13
data8 0x9BC67404F348C0CE, 0x00003FDB //A12 = 1.7709590771868743572061278273e-11
data8 0xF4B5D0348AFAAC7A, 0x00003FDB //A11 = 2.7820329729584630464848160970e-11
data8 0x83AB447FF619DA4A, 0x0000BFE2 //A10 = -1.9160363295631539615395477207e-09
data8 0x82115AB487202E7B, 0x00003FE0 //A9 = 4.7318386460142606822119637959e-10
data8 0xB84D5B0AE17054AA, 0x00003FE8 //A8 = 1.7164477188916895004843908951e-07
data8 0xB2E085C1C4AA06E5, 0x0000BFE9 //A7 = -3.3318445266863554512523957574e-07
data8 0xCD3CA2E6C3971666, 0x0000BFEE //A6 = -1.2233070175554502732980949519e-05
data8 0xBA445C53F8DD40E6, 0x00003FF0 //A5 = 4.4409521535330413551781808621e-05
data8 0xAA94D5E68033B764, 0x00003FF4 //A4 = 6.5071635765452563856926608000e-04
LOCAL_OBJECT_END(_0p25_to_0p5_data_tail)

LOCAL_OBJECT_START(_0p5_to_1_data_tail)
// Polynomial coefficients for the erf(x), 1/2 <= |x| < 1
data8 0x9ED99EDF111CB785, 0x0000BFE4 //A13 = -9.2462916180079278241704711522e-09
data8 0xDEAF7539AE2FB062, 0x0000BFE5 //A12 = -2.5923990465973151101298441139e-08
data8 0xA392D5E5CC9DB1A7, 0x00003FE9 //A11 = 3.0467952847327075747032372101e-07
data8 0xC311A7619B96CA1A, 0x00003FE8 //A10 = 1.8167212632079596881709988649e-07
data8 0x82082E6B6A93F116, 0x0000BFEE //A9 = -7.7505086843257228386931766018e-06
data8 0x96D9997CF326A36D, 0x00003FEE //A8 = 8.9913605625817479172071008270e-06
data8 0x97057D85DCB0ED99, 0x00003FF2 //A7 = 1.4402527482741758767786898553e-04
data8 0xDC23BCB3599C0490, 0x0000BFF3 //A6 = -4.1988296144950673955519083419e-04
data8 0xDA150C4867208A81, 0x0000BFF5 //A5 = -1.6638352864915033417887831090e-03
data8 0x9A4DAF550A2CC29A, 0x00003FF8 //A4 = 9.4179355839141698591817907680e-03
LOCAL_OBJECT_END(_0p5_to_1_data_tail)

LOCAL_OBJECT_START(_1_to_2_data_tail)
// Polynomial coefficients for the erf(x), 1 <= |x| < 2.0
data8 0x969EAC5C7B46CAB9, 0x00003FEF //A13 = 1.7955281439310148162059582795e-05
data8 0xA2ED832912E9FCD9, 0x00003FF1 //A12 = 7.7690020847111408916570845775e-05
data8 0x85677C39C48E43E7, 0x0000BFF3 //A11 = -2.5444839340796031538582511806e-04
data8 0xC2DAFA91683DAAE4, 0x0000BFF1 //A10 = -9.2914288456063075386925076097e-05
data8 0xE01C061CBC6A2825, 0x00003FF5 //A9 = 1.7098195515864039518892834211e-03
data8 0x9AD7271CAFD01C78, 0x0000BFF6 //A8 = -2.3626776207372761518718893636e-03
data8 0x9B6B9D30EDD5F4FF, 0x0000BFF7 //A7 = -4.7430532011804570628999212874e-03
data8 0x9E51EB9623F1D446, 0x00003FF9 //A6 = 1.9326171998839772791190405201e-02
data8 0xF391B935C12546DE, 0x0000BFF8 //A5 = -1.4866286152953671441682166195e-02
data8 0xB6AD4AE850DBF526, 0x0000BFFA //A4 = -4.4598858458861014323191919669e-02
LOCAL_OBJECT_END(_1_to_2_data_tail)

LOCAL_OBJECT_START(_2_to_3p25_data_tail)
// Polynomial coefficients for the erf(x), 2 <= |x| < 3.25
data8 0x847C24DAC7C7558B, 0x00003FF5 //A13 = 1.0107798565424606512130100541e-03
data8 0xCB6340EAF02C3DF8, 0x00003FF8 //A12 = 1.2413800617425931997420375435e-02
data8 0xB5163D252DBBC107, 0x0000BFF9 //A11 = -2.2105330871844825370020459523e-02
data8 0x82FF9C0B68E331E4, 0x00003FF9 //A10 = 1.5991024756001692140897408128e-02
data8 0xE9519E4A49752E04, 0x00003FF7 //A9 = 7.1203253651891723548763348088e-03
data8 0x8D52F11B7AE846D9, 0x0000BFFA //A8 = -3.4502927613795425888684181521e-02
data8 0xCCC5A3E32BC6FA30, 0x00003FFA //A7 = 4.9993171868423886228679106871e-02
data8 0xC1791AD8284A1919, 0x0000BFFA //A6 = -4.7234635220336795411997070641e-02
data8 0x853DAAA35A8A3C18, 0x00003FFA //A5 = 3.2529512934760303976755163452e-02
data8 0x88E42D8F47FAB60E, 0x0000BFF9 //A4 = -1.6710366233609742619461063050e-02
LOCAL_OBJECT_END(_2_to_3p25_data_tail)

LOCAL_OBJECT_START(_4_to_6p53_data_tail)
// Polynomial coefficients for the erf(x), 4 <= |x| < 6.53
data8 0xD8235ABF08B8A6D1, 0x00003FEE //A13 = 1.2882834877224764938429832586e-05
data8 0xAEDF44F9C77844C2, 0x0000BFEC //A12 = -2.6057980393716019511497492890e-06
data8 0xCCD5490956A4FCFD, 0x00003FEA //A11 = 7.6306293047300300284923464089e-07
data8 0xF71AF0126EE26AEA, 0x0000BFE8 //A10 = -2.3013467500738417953513680935e-07
data8 0xE4CE68089858AC20, 0x00003FE6 //A9 = 5.3273112263151109935867439775e-08
data8 0xBD15106FBBAEE593, 0x0000BFE4 //A8 = -1.1006037358336556244645388790e-08
data8 0x8BBF9A5769B6E480, 0x00003FE2 //A7 = 2.0336075804332107927300019116e-09
data8 0xB049D845D105E302, 0x0000BFDF //A6 = -3.2066683399502826067820249320e-10
data8 0xBAC69B3F0DFE5483, 0x00003FDC //A5 = 4.2467901578369360007795282687e-11
data8 0xA29C398F83F8A0D1, 0x0000BFD9 //A4 = -4.6216613698438694005327544047e-12
LOCAL_OBJECT_END(_4_to_6p53_data_tail)

LOCAL_OBJECT_START(_3p25_to_4_data_tail)
// Polynomial coefficients for the erf(x), 3.25 <= |x| < 4
data8 0x95BE1BEAD738160F, 0x00003FF2 //A13 = 1.4280568455209843005829620687e-04
data8 0x8108C8FFAC0F0B21, 0x0000BFF4 //A12 = -4.9222685622046459346377033307e-04
data8 0xD72A7FAEE7832BBE, 0x00003FF4 //A11 = 8.2079319302109644436194651098e-04
data8 0x823AB4281CA7BBE7, 0x0000BFF5 //A10 = -9.9357079675971109178261577703e-04
data8 0xFA1232D476048D11, 0x00003FF4 //A9 = 9.5394549599882496825916138915e-04
data8 0xC463D7AF88025FB2, 0x0000BFF4 //A8 = -7.4916843357898101689031755368e-04
data8 0xFEBE32B6B379D072, 0x00003FF3 //A7 = 4.8588363901002111193445057206e-04
data8 0x882829BB68409BF3, 0x0000BFF3 //A6 = -2.5969865184916169002074135516e-04
data8 0xED2F886E29DAAB09, 0x00003FF1 //A5 = 1.1309894347742479284610149994e-04
data8 0xA4C07129436555B2, 0x0000BFF0 //A4 = -3.9279872584973887163830479579e-05
LOCAL_OBJECT_END(_3p25_to_4_data_tail)


LOCAL_OBJECT_START(_0_to_1o8_data)
// Polynomial coefficients for the erf(x), 0.0 <= |x| < 0.125
data8 0x3FF20DD750429B6D, 0x3C71AE3A8DDFFEDE //A1H, A1L
data8 0xF8B0DACE42525CC2, 0x0000BFEE //A15
data8 0xFCD02E1BF0EC2C37, 0x00003FF1 //A13
data8 0xE016D968FE473B5E, 0x0000BFF4 //A11
data8 0xAB2DE68711BF5A79, 0x00003FF7 //A9
data8 0xDC16718944518309, 0x0000BFF9 //A7
data8 0xE71790D0215F0C8F, 0x00003FFB //A5
data8 0xC093A3581BCF3612, 0x0000BFFD //A3
LOCAL_OBJECT_END(_0_to_1o8_data)


LOCAL_OBJECT_START(_denorm_data)
data8 0x3FF20DD750429B6D //A1H = 1.1283791670955125585606992900e+00
data8 0x3C71AE3A914FED80 //A1L = 1.5335459613165880745599768129e-17
LOCAL_OBJECT_END(_denorm_data)


.section .text
GLOBAL_LIBM_ENTRY(erfl)

{ .mfi
      alloc          r32         = ar.pfs, 0, 21, 0, 0
      fmerge.se      fArgAbsNorm = f1, f8      // normalized x (1.0 <= x < 2.0)
      addl           rSignBit    = 0x20000, r0 // Set sign bit for exponent
}
{ .mlx
      addl           rDataPtr    = @ltoff(erfl_data), gp // Get common data ptr
      movl           r1p5        = 0x3FF8000000000000    // 1.5 in dbl repres.
};;

{ .mfi
      getf.exp       rArgExp     = f8              // Get arg exponent
      fclass.m       p6,p0       = f8, 0xEF // Filter 0, denormals and specials
                            // 0xEF = @qnan|@snan|@pos|@neg|@zero|@unorm|@inf
      addl           rBias       = 0xfffc, r0 // Value to subtract from exp
                                              // to get actual interval number
}
{ .mfi
      ld8            rDataPtr    = [rDataPtr]  // Get real common data pointer
      fma.s1         fArgSqr     = f8, f8, f0  // x^2 (for [0;1/8] path)
      addl           r2to4       = 0x10000, r0 // unbiased exponent
                                               // for [2;4] binary interval
};;

{ .mfi
      getf.sig       rArgSig     = f8              // Get arg significand
      fcmp.lt.s1     p15, p14    = f8, f0          // Is arg negative/positive?
      addl           rSaturation = 0xd0e, r0       // First 12 bits of
                                                   // saturation value signif.
}
{ .mfi
      setf.d         f1p5        = r1p5            // 1.5 construction
      fma.s1         f2p0        = f1,f1,f1        // 2.0 construction
      addl           r3p25Sign   = 0xd00, r0       // First 12 bits of
                                                   // 3.25 value signif.
};;

{ .mfi
      addl           rTailDataPtr = 0x700, rDataPtr  // Pointer to "tail" data
      nop.f          0
      andcm          rArgExp     = rArgExp, rSignBit // Remove sign of exp
}
{ .mfb
      addl           rTiny       = 0xf000, r0 // Tiny value for saturation path
      nop.f          0
(p6)  br.cond.spnt   erfl_spec              // Branch to zero, denorm & specs
};;

{ .mfi
      sub            rInterval   = rArgExp, rBias // Get actual interval number
      nop.f          0
      shr.u          rArgSig     = rArgSig, 52    // Leave only 12 bits of sign.
}
{ .mfi
      adds           rShiftedDataPtr = 0x10, rDataPtr // Second ptr to data
      nop.f          0
      cmp.eq         p8, p10     = r2to4, rArgExp // If exp is in 2to4 interval?
};;

{ .mfi
(p8)  cmp.le         p8, p10     = r3p25Sign, rArgSig // If sign. is greater
                            //  than 1.25? (means arg is in [3.25;4] interval)
      nop.f          0
      shl            rOffset     = rInterval, 8 // Make offset from
                                                // interval number
}
{ .mfi
      cmp.gt         p9, p0      = 0x0, rInterval // If interval is less than 0
                                                  // (means arg is in [0; 1/8])
      nop.f          0
      cmp.eq         p7, p0      = 0x5, rInterval // If arg is in [4:8] interv.?
};;

{ .mfi
(p8)  adds           rOffset     = 0x200, rOffset // Add additional offset
                                 // if arg is in [3.25;4] (another data set)
      fma.s1         fArgCube    = fArgSqr, f8, f0  // x^3 (for [0;1/8] path)
      shl            rTailOffset = rInterval, 7  // Make offset to "tail" data
                                                 // from interval number
}
{ .mib
      setf.exp       fTiny       = rTiny // Construct "tiny" value
                                         // for saturation path
      cmp.ltu        p11, p0     = 0x5, rInterval // if arg > 8
(p9)  br.cond.spnt   _0_to_1o8
};;

{ .mfi
      add            rAddr1      = rDataPtr, rOffset // Get address for
                                                     // interval data
      nop.f          0
      shl            rTailAddOffset = rInterval, 5 // Offset to interval
                                                   // "tail" data
}
{ .mib
      add            rAddr2      = rShiftedDataPtr, rOffset // Get second
                                                 // address for interval data
(p7)  cmp.leu        p11, p0     = rSaturation, rArgSig // if arg is
                                                        // in [6.53;8] interval
(p11) br.cond.spnt   _saturation // Branch to Saturation path
};;

{ .mmi
      ldfe           fA3         = [rAddr1], 0x90 // Load A3
      ldfpd          fA2H, fA2L  = [rAddr2], 16 // Load A2High, A2Low
      add            rTailOffset = rTailOffset, rTailAddOffset // "Tail" offset
};;

{ .mmi
      ldfe           fA20        = [rAddr1], 16 // Load A20
      ldfpd          fA1H, fA1L  = [rAddr2], 16 // Load A1High, A1Low
(p8)  adds           rTailOffset = 0x140, rTailOffset // Additional offset
                                                      //  for [3.24;4] interval
};;

{ .mmi
      ldfe           fA19        = [rAddr1], 16 // Load A19
      ldfpd          fA0H, fA0L  = [rAddr2], 16 // Load A0High, A0Low
      add            rTailAddr1  = rTailDataPtr, rTailOffset // First tail
                                                             // data address
};;

.pred.rel "mutex",p8,p10
{ .mfi
      ldfe           fA18        = [rAddr1], 16 // Load A18
(p8)  fms.s1         fArgAbsNorm = fArgAbsNorm, f1, f2p0 // Add 2.0
                          // to normalized arg (for [3.24;4] interval)
      adds           rTailAddr2  = 0x10, rTailAddr1  // First tail
                                                     // data address
}
{ .mfi
      ldfe           fA25        = [rAddr2], 16 // Load A25
(p10) fms.s1         fArgAbsNorm = fArgAbsNorm, f1, f1p5  // Add 1.5
                                                // to normalized arg
      nop.i          0
};;

{ .mmi
      ldfe           fA17        = [rAddr1], 16 // Load A17
      ldfe           fA24        = [rAddr2], 16 // Load A24
      nop.i          0
};;

{ .mmi
      ldfe           fA16        = [rAddr1], 16 // Load A16
      ldfe           fA23        = [rAddr2], 16 // Load A23
      nop.i          0
};;

{ .mmi
      ldfe           fA15        = [rAddr1], 16 // Load A15
      ldfe           fA22        = [rAddr2], 16 // Load A22
      nop.i          0
};;

{ .mmi
      ldfe           fA14        = [rAddr1], 16 // Load A14
      ldfe           fA21        = [rAddr2], 16 // Load A21
      nop.i          0
};;

{ .mfi
      ldfe           fA13        = [rTailAddr1], 32              // Load A13
      fms.s1         fArgAbsNorm2 = fArgAbsNorm, fArgAbsNorm, f0 // x^2
      nop.i          0
}
{ .mfi
      ldfe           fA12        = [rTailAddr2], 32 // Load A12
      nop.f          0
      nop.i          0
};;

{ .mfi
      ldfe           fA11        = [rTailAddr1], 32       // Load A11
      fma.s1         fRes3H      = fA3, fArgAbsNorm, fA2H // (A3*x+A2)*x^2
      nop.i          0
}
{ .mfi
      ldfe           fA10        = [rTailAddr2], 32     // Load A10
      fma.s1         fTH         = fA3, fArgAbsNorm, f0 // (A3*x+A2)*x^2
      nop.i          0
};;

{ .mfi
      ldfe           fA9         = [rTailAddr1], 32 // Load A9
      fma.s1         fTT2        = fA1L, fArgAbsNorm, f0 // A1*x+A0
      nop.i          0
}
{ .mfi
      ldfe           fA8         = [rTailAddr2], 32 // Load A8
      nop.f          0
      nop.i          0
};;

{ .mmi
      ldfe           fA7         = [rTailAddr1], 32 // Load A7
      ldfe           fA6         = [rTailAddr2], 32 // Load A6
      nop.i          0
};;

{ .mmi
      ldfe           fA5         = [rTailAddr1], 32 // Load A5
      ldfe           fA4         = [rTailAddr2], 32 // Load A4
      nop.i          0
};;

{ .mfi
      nop.m          0
      fms.s1         fArgAbsNorm2L = fArgAbsNorm, fArgAbsNorm, fArgAbsNorm2
                                                    // Low part of x^2 (delta)
      nop.i          0
}
{ .mfi
      nop.m          0
      fms.s1         fArgAbsNorm4  = fArgAbsNorm2, fArgAbsNorm2, f0 // x^4
      nop.i          0
};;

{ .mfi
      nop.m          0
      fms.s1         fRes3L      = fA2H, f1, fRes3H // // (A3*x+A2)*x^2
      nop.i          0
};;

{ .mfi
      nop.m          0
      fms.s1         fArgAbsNorm3 = fArgAbsNorm2, fArgAbsNorm, f0 // x^3
      nop.i          0
}
{ .mfi
      nop.m          0
      fma.s1         fTH2        = fA1H, fArgAbsNorm, fTT2 // A1*x+A0
      nop.i          0
};;

{ .mfi
      nop.m          0
      fma.s1         fA23        = fA24,  fArgAbsNorm, fA23 // Polynomial tail
      nop.i          0
}
{ .mfi
      nop.m          0
      fma.s1         fA21        = fA22,  fArgAbsNorm, fA21 // Polynomial tail
      nop.i          0
};;

{ .mfi
      nop.m          0
      fma.s1         fA12        = fA13,  fArgAbsNorm, fA12 // Polynomial tail
      nop.i          0
}
;;

{ .mfi
      nop.m          0
      fma.s1         fRes3L      = fRes3L, f1, fTH // (A3*x+A2)*x^2
      nop.i          0
}
{ .mfi
      nop.m          0
      fma.s1         fA19        = fA20,  fArgAbsNorm, fA19 // Polynomial tail
      nop.i          0
};;

{ .mfi
      nop.m          0
      fma.s1         fRes1H      = fTH2, f1, fA0H // A1*x+A0
      nop.i          0
}
{ .mfi
      nop.m          0
      fms.s1         fTL2        = fA1H, fArgAbsNorm, fTH2 // A1*x+A0
      nop.i          0
};;

{ .mfi
      nop.m          0
      fma.s1         fA8         = fA9,  fArgAbsNorm, fA8 // Polynomial tail
      nop.i          0
}
{ .mfi
      nop.m          0
      fma.s1         fA10        = fA11,  fArgAbsNorm, fA10 // Polynomial tail
      nop.i          0
};;
{ .mfi
      nop.m          0
      fma.s1         fA15        = fA16,  fArgAbsNorm, fA15 // Polynomial tail
      nop.i          0
}
{ .mfi
      nop.m          0
      fma.s1         fA17        = fA18,  fArgAbsNorm, fA17 // Polynomial tail
      nop.i          0
};;
{ .mfi
      nop.m          0
      fms.s1         fArgAbsNorm11 = fArgAbsNorm4, fArgAbsNorm4, f0 // x^8
      nop.i          0
}
{ .mfi
      nop.m          0
      fma.s1         fA4         = fA5,  fArgAbsNorm, fA4 // Polynomial tail
      nop.i          0
};;

{ .mfi
      nop.m          0
      fma.s1         fRes3L      = fRes3L, f1, fA2L // (A3*x+A2)*x^2
      nop.i          0
}
{ .mfi
      nop.m          0
      fma.s1         fA6         = fA7,  fArgAbsNorm, fA6 // Polynomial tail
      nop.i          0
};;

{ .mfi
      nop.m          0
      fma.s1         fTL2        = fTL2, f1, fTT2 // A1*x+A0
      nop.i          0
}
{ .mfi
      nop.m          0
      fms.s1         fRes1L      = fA0H, f1, fRes1H // A1*x+A0
      nop.i          0
};;

{ .mfi
      nop.m          0
      fma.s1         fA23        = fA25,  fArgAbsNorm2, fA23 // Polynomial tail
      nop.i          0
}
{ .mfi
      nop.m          0
      fma.s1         fA12        = fA14,  fArgAbsNorm2, fA12 // Polynomial tail
      nop.i          0
};;

{ .mfi
      nop.m          0
      fma.s1         fA19        = fA21,  fArgAbsNorm2, fA19  // Polynomial tail
      nop.i          0
}
{ .mfi
      nop.m          0
      fma.s1         fA8         = fA10,  fArgAbsNorm2, fA8 // Polynomial tail
      nop.i          0
};;

{ .mfi
      nop.m          0
      fma.s1         fA15        = fA17,  fArgAbsNorm2, fA15 // Polynomial tail
      nop.i          0
}
{ .mfi
      nop.m          0
      fms.s1         fArgAbsNorm11 = fArgAbsNorm11, fArgAbsNorm3, f0 // x^11
      nop.i          0
};;

{ .mfi
      nop.m          0
      fma.s1         fTT         = fRes3L, fArgAbsNorm2, f0 // (A3*x+A2)*x^2
      nop.i          0
}
{ .mfi
      nop.m          0
      fma.s1         fA4         = fA6,  fArgAbsNorm2, fA4 // Polynomial tail
      nop.i          0
};;

{ .mfi
      nop.m          0
      fma.s1         fRes1L      = fRes1L, f1, fTH2 // A1*x+A0
      nop.i          0
};;

{ .mfi
      nop.m          0
      fma.s1         fA19        = fA23,  fArgAbsNorm4, fA19 // Polynomial tail
      nop.i          0
}
{ .mfi
      nop.m          0
      fma.s1         fA8         = fA12,  fArgAbsNorm4, fA8 // Polynomial tail
      nop.i          0
};;

{ .mfi
      nop.m          0
      fma.s1         fTT         = fRes3H, fArgAbsNorm2L, fTT // (A3*x+A2)*x^2
      nop.i          0
};;

{ .mfi
      nop.m          0
      fma.s1         fRes1L      = fRes1L, f1, fTL2 // A1*x+A0
      nop.i          0
};;

{ .mfi
      nop.m          0
      fma.s1         fA15        = fA19,  fArgAbsNorm4, fA15 // Polynomial tail
      nop.i          0
}
{ .mfi
      nop.m          0
      fma.s1         fA4         = fA8,  fArgAbsNorm4, fA4 // Polynomial tail
      nop.i          0
};;

{ .mfi
      nop.m          0
      fma.s1         fRes2H      = fRes3H, fArgAbsNorm2, fTT // (A3*x+A2)*x^2
      nop.i          0
};;

{ .mfi
      nop.m          0
      fma.s1         fRes1L      = fRes1L, f1, fA0L // A1*x+A0
      nop.i          0
};;

{ .mfi
      nop.m          0
      fma.s1         fRes4       = fA15, fArgAbsNorm11, fA4 // Result of
                                                      // polynomial tail
      nop.i          0
};;

{ .mfi
      nop.m          0
      fms.s1         fRes2L      = fRes3H, fArgAbsNorm2, fRes2H // (A3*x+A2)*x^2
      nop.i          0
}
{ .mfi
      nop.m          0
      fma.s1         fResH       = fRes2H, f1, fRes1H // High result
      nop.i          0
};;

{ .mfi
      nop.m          0
      fma.s1         fRes1L      = fRes4, fArgAbsNorm4, fRes1L // A1*x+A0
      nop.i          0
};;

{ .mfi
      nop.m          0
      fma.s1         fRes2L      = fRes2L, f1, fTT // (A3*x+A2)*x^2
      nop.i          0
}
{ .mfi
      nop.m          0
      fms.s1         fResL       = fRes1H, f1, fResH // Low result
      nop.i          0
};;

{ .mfi
      nop.m          0
      fma.s1         fRes1L      = fRes1L, f1, fRes2L // Low result
      nop.i          0
}
{ .mfi
      nop.m          0
      fma.s1         fResL       = fResL, f1, fRes2H // Low result
      nop.i          0
};;

{ .mfi
      nop.m          0
(p15) fneg           fResH       = fResH // Invert high result if arg is neg.
      nop.i          0
};;

{ .mfi
      nop.m          0
      fma.s1         fResL       = fResL, f1, fRes1L // Low result
      nop.i          0
};;

.pred.rel "mutex",p14,p15
{ .mfi
      nop.m          0
(p14) fma.s0         f8          = fResH, f1, fResL // Add high and low results
      nop.i          0
}
{ .mfb
      nop.m          0
(p15) fms.s0         f8          = fResH, f1, fResL // Add high and low results
      br.ret.sptk    b0          // Main path return
};;

//  satiration path ////////////////////////////////////////////////////////////
_saturation:

.pred.rel "mutex",p14,p15
{ .mfi
      nop.m          0
(p14) fms.s0            f8          = f1, f1, fTiny // Saturation result r = 1-tiny
      nop.i 0
};;
{ .mfb
      nop.m          0
(p15) fnma.s0           f8          = f1, f1, fTiny // Saturation result r = tiny-1
      br.ret.sptk    b0         // Saturation path return
};;


//  0, denormals and special IEEE numbers path /////////////////////////////////
erfl_spec:

{ .mfi
      addl           rDataPtr    = 0xBE0, rDataPtr // Ptr to denormals coeffs
      fclass.m       p6,p0       = f8, 0x23 // To filter infinities
                                          // 0x23 = @pos|@neg|@inf
      nop.i          0
};;

{ .mfi
      ldfpd          fA1H, fA1L  = [rDataPtr] // Load denormals coeffs A1H, A1L
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
(p7)  fma.s0         f8          = f8, f1, f8    // +/-0 for 0 args
                                                 // and NaNs for NaNs
(p7)  br.ret.spnt    b0                          // exit for x = NaN or +/-0
};;

{ .mfi
      nop.m          0
      fnorm.s0       f8          = f8            // Normalize arg
      nop.i          0
};;

{ .mfi
      nop.m          0
      fms.s1         fRes1H      = f8, fA1H, f0   // HighRes
      nop.i          0
}
{ .mfi
      nop.m          0
      fms.s1         fRes1L      = f8, fA1L, f0   // LowRes
      nop.i          0
};;

{ .mfi
      nop.m          0
      fms.s1         fRes1Hd     = f8, fA1H, fRes1H // HighRes delta
      nop.i          0
};;

{ .mfi
      nop.m          0
      fma.s1         fRes        = fRes1L, f1,  fRes1Hd // LowRes+HighRes delta
      nop.i          0
};;

{ .mfi
      nop.m          0
      fma.s1         fRes        = f8, f8, fRes // r=x^2+r
      nop.i          0
};;

{ .mfb
      nop.m          0
      fma.s0         f8          = fRes, f1, fRes1H  // res = r+ResHigh
      br.ret.sptk    b0          // 0, denormals, specials return
};;


//  0 < |x| < 1/8 path /////////////////////////////////////////////////////////
_0_to_1o8:

{ .mmi
      adds           rAddr1      = 0xB60, rDataPtr // Ptr 1 to coeffs
      adds           rAddr2      = 0xB80, rDataPtr // Ptr 2 to coeffs
      nop.i          0
};;

{ .mmi
      ldfpd          fA1H, fA1L  = [rAddr1], 16 // Load A1High, A1Low
      ldfe           fA13        = [rAddr2], 16 // Load A13
      nop.i          0
};;

{ .mmi
      ldfe           fA15        = [rAddr1], 48 // Load A15
      ldfe           fA11        = [rAddr2], 32 // Load A11
      nop.i          0
};;

{ .mmi
      ldfe           fA9         = [rAddr1], 32 // Load A9
      ldfe           fA7         = [rAddr2], 32 // Load A7
      nop.i          0
};;

{ .mmi
      ldfe           fA5         = [rAddr1]  // Load A5
      ldfe           fA3         = [rAddr2] // Load A3
      nop.i          0
};;

{ .mfi
      nop.m          0
      fms.s1         fRes1H      = f8, fA1H, f0 // x*(A1H+A1L)
      nop.i          0
}
{ .mfi
      nop.m          0
      fms.s1         fRes1L      = f8, fA1L, f0 // x*(A1H+A1L)
      nop.i          0
};;

{ .mfi
      nop.m          0
      fma.s1         fA11        = fA13, fArgSqr, fA11 // Polynomial tail
      nop.i          0
}
{ .mfi
      nop.m          0
      fma.s1         fArgFour    = fArgSqr, fArgSqr, f0 // a^4
      nop.i          0
};;


{ .mfi
      nop.m          0
      fma.s1         fA3         = fA5, fArgSqr, fA3 // Polynomial tail
      nop.i          0
}
{ .mfi
      nop.m          0
      fma.s1         fA7         = fA9, fArgSqr, fA7 // Polynomial tail
      nop.i          0
};;


{ .mfi
      nop.m          0
      fms.s1         fRes1Hd     = f8, fA1H, fRes1H // x*(A1H+A1L) delta
      nop.i          0
};;

{ .mfi
      nop.m          0
      fma.s1         fA11        = fA15, fArgFour, fA11 // Polynomial tail
      nop.i          0
};;

{ .mfi
      nop.m          0
      fma.s1         fA3         = fA7, fArgFour, fA3 // Polynomial tail
      nop.i          0
}
{ .mfi
      nop.m          0
      fma.s1         fArgEight   = fArgFour, fArgFour, f0 // a^8
      nop.i          0
};;

{ .mfi
      nop.m          0
      fma.s1         f8          = fRes1L, f1,  fRes1Hd // x*(A1H+A1L)
      nop.i          0
};;

{ .mfi
      nop.m          0
      fma.s1         fRes        = fA11, fArgEight, fA3 //Polynomial tail result
      nop.i          0
};;

{ .mfi
      nop.m          0
      fma.s1         f8          = fRes, fArgCube, f8 // (Polynomial tail)*x^3
      nop.i          0
};;

{ .mfb
      nop.m          0
      fma.s0         f8          = f8, f1, fRes1H  // (Polynomial tail)*x^3 +
                                                   // + x*(A1H+A1L)
      br.ret.sptk    b0          // [0;1/8] interval return
};;


GLOBAL_LIBM_END(erfl)
libm_alias_ldouble_other (erf, erf)
