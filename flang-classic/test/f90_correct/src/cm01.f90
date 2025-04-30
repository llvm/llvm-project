!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

!
! Most trivial of tests to (try to) exercise all the transcendental intrinsics
! recognized by the Fortran compilers.
! No attempt is made to test all quadrants of the real and complex
! sine/cosine/tangent functions.
!
! Software (test) vector lengths have been chosen to try and exercise all
! supported hardware vector lengths.
!
! Correctness is not exhaustive and only good to the relative error defined
! in module real_diff_mod and complex_diff_mod below.
!
! REAL(KIND=8) use the reference values from the REAL(KIND=4) calls.
! COMPLEX(KIND=8) use the reference values from the COMPLEX(KIND=4) calls.
! Arguments for all intrinsics are the same.
! 
!
module complex_data_mod
  implicit none
  integer, parameter :: ivl = 15        ! 8 + 4 + 2 + 1
  complex(kind=4), dimension(ivl) :: carg1
  complex(kind=4), dimension(ivl) :: carg2
  complex(kind=8), dimension(ivl) :: zarg1
  complex(kind=8), dimension(ivl) :: zarg2

  !
  ! Do not make reference values constants (PARAMETER). Compiler
  ! will optimize too many of the call out and replace with compile
  ! time constants.
  !
  
  ! cabs does not return a complex, but keep structure consistent.
  complex(kind=4), dimension(ivl) :: cabs_ref = (/ &
     (14.14214,0.000000),  (15.29706,0.000000),  (16.49242,0.000000), &
     (17.72005,0.000000),  (18.97367,0.000000),  (20.24846,0.000000), &
     (21.54066,0.000000),  (22.84732,0.000000),  (24.16609,0.000000), &
     (25.49510,0.000000),  (26.83282,0.000000),  (28.17801,0.000000), &
     (29.52965,0.000000),  (30.88689,0.000000),  (32.24903,0.000000) &
  /)
  complex(kind=4), dimension(ivl) :: cacos_ref = (/ &
     (1.429246,-3.343505),  (1.373812,-3.421789),  (1.326253,-3.496886), &
     (1.285168,-3.568568),  (1.249464,-3.636735),  (1.218194,-3.701709), &
     (1.190664,-3.763532),  (1.166254,-3.822322),  (1.144478,-3.878369), &
     (1.124987,-3.931868),  (1.107423,-3.982925),  (1.091521,-4.031925), &
     (1.077114,-4.078710),  (1.063921,-4.123626),  (1.051879,-4.166727) &
  /)
  complex(kind=4), dimension(ivl) :: casin_ref = (/ &
     (0.1415500,3.343505),  (0.1969846,3.421789),  (0.2445434,3.496886), &
     (0.2856286,3.568568),  (0.3213326,3.636735),  (0.3526020,3.701709), &
     (0.3801323,3.763532),  (0.4045426,3.822322),  (0.4263184,3.878369), &
     (0.4458098,3.931868),  (0.4633729,3.982925),  (0.4792751,4.031925), &
     (0.4936819,4.078710),  (0.5068753,4.123626),  (0.5189172,4.166727) &
  /)
  complex(kind=4), dimension(ivl) :: catan_ref = (/ &
     (1.560747,7.0107609E-02),  (1.557924,6.4179957E-02),  (1.556040,5.8878709E-02), &
     (1.554827,5.4179303E-02),  (1.554089,5.0027765E-02),  (1.553688,4.6361119E-02), &
     (1.553525,4.3117315E-02),  (1.553529,4.0239606E-02),  (1.553650,3.7678007E-02), &
     (1.553854,3.5389237E-02),  (1.554113,3.3336412E-02),  (1.554409,3.1488113E-02), &
     (1.554728,2.9817654E-02),  (1.555062,2.8302448E-02),  (1.555402,2.6923217E-02) &
  /)
  complex(kind=4), dimension(ivl) :: ccos_ref = (/ &
     (-250230.0,-546762.4),            (-1618151.,-230661.9),            (-2904175.,3362516.),           &
     (3425924.,1.1581385E+07),         (3.1522374E+07,9173206.),         (6.7279104E+07,-5.8630240E+07), &
     (-3.5295776E+07,-2.4000109E+08),  (-6.0080646E+08,-2.7175418E+08),  (-1.5039991E+09,9.7513421E+08), &
     (2.1563780E+07,4.8723543E+09),    (1.1176476E+10,7.1066732E+09),    (3.2670310E+10,-1.5127043E+10), &
     (1.3381760E+10,-9.6945594E+10),   (-2.0209531E+11,-1.7299227E+11),  (-6.9251092E+11,2.0819109E+11)  &
  /)
  complex(kind=4), dimension(ivl) :: csin_ref = (/ &
     (546762.4,-250230.0),            (230661.9,-1618151.),            (-3362516.,-2904175.),           &
     (-1.1581385E+07,3425924.),       (-9173206.,3.1522374E+07),       (5.8630240E+07,6.7279104E+07),   &
     (2.4000109E+08,-3.5295776E+07),  (2.7175418E+08,-6.0080646E+08),  (-9.7513421E+08,-1.5039991E+09), &
     (-4.8723543E+09,2.1563780E+07),  (-7.1066732E+09,1.1176476E+10),  (1.5127043E+10,3.2670310E+10),   &
     (9.6945594E+10,1.3381760E+10),   (1.7299227E+11,-2.0209531E+11),  (-2.0819109E+11,-6.9251092E+11)  &
  /)
  complex(kind=4), dimension(ivl) :: ctan_ref = (/ &
     (-1.0465671E-12,1.000000),  (-5.2293293E-14,0.9999999),  (2.5058792E-14,1.000000), &
     (-1.8648049E-15,1.000000),  (-2.4891860E-16,1.000000),   (6.2192957E-17,1.000000), &
     (-2.4462307E-18,1.000000),  (-8.6356353E-19,1.000000),   (1.4207495E-19,1.000000), &
     (-1.8641948E-22,1.000000),  (-2.5811955E-21,1.000000),   (2.9415687E-22,1.000000), &
     (1.4142796E-23,1.000000),   (-6.9806972E-24,1.000000),   (5.2726230E-25,1.000000) &
  /)
  complex(kind=4), dimension(ivl) :: ccosh_ref = (/ &
     (0.5144322,3.592795),   (-7.648281,6.514503),  (-26.15199,-7.856858), &
     (-20.41986,-71.33878),  (133.1962,-151.4840),  (542.1236,82.18009),   &
     (608.2378,1360.726),    (-2219.148,3389.745),  (-11012.80,-97.48153), &
     (-15951.46,-25333.36),  (34518.59,-73693.61),  (219260.7,-29277.09),  &
     (388994.0,458528.0),    (-477503.4,1563205.),  (-4276912.,1203650.)   &
  /)
  complex(kind=4), dimension(ivl) :: csinh_ref = (/ &
     (0.4959268,3.726859),   (-7.610458,6.546879),  (-26.13445,-7.862130), &
     (-20.41800,-71.34526),  (133.1946,-151.4859),  (542.1227,82.18022),   &
     (608.2377,1360.726),    (-2219.148,3389.745),  (-11012.80,-97.48153), &
     (-15951.46,-25333.36),  (34518.59,-73693.61),  (219260.7,-29277.10),  &
     (388994.0,458528.0),    (-477503.4,1563205.),  (-4276912.,1203650.)   &
  /)
  complex(kind=4), dimension(ivl) :: ctanh_ref = (/ &
     (1.035842,1.0282761E-02),  (0.9992237,-4.8943986E-03),  (0.9994403,3.6975901E-04), &
     (1.000077,4.8044341E-05),  (1.000002,-1.2187418E-05),   (0.9999983,4.9287712E-07), &
     (1.000000,1.6770289E-07),  (1.000000,-2.7917206E-08),   (1.000000,7.2972781E-11),  &
     (1.000000,5.0310200E-10),  (1.000000,-5.8005292E-11),   (1.000000,-2.6809934E-12), & 
     (1.000000,1.3643877E-12),  (0.9999999,-1.0457874E-13),  (1.000000,-1.3210016E-14)  &
  /)
  complex(kind=4), dimension(ivl) :: cdiv_ref = (/ &
     (0.2302071,8.8915959E-02),  (0.2475305,8.1929117E-02),  (0.2641509,7.5471699E-02), &
     (0.2801061,6.9496021E-02),  (0.2954315,6.3959390E-02),  (0.3101604,5.8823533E-02), &
     (0.3243243,5.4054063E-02),  (0.3379526,4.9620021E-02),  (0.3510730,4.5493562E-02), &
     (0.3637114,4.1649491E-02),  (0.3758922,3.8065027E-02),  (0.3876383,3.4719579E-02), &
     (0.3989713,3.1594407E-02),  (0.4099115,2.8672563E-02),  (0.4204778,2.5938567E-02)  &
  /)
  complex(kind=4), dimension(ivl) :: cdivr_ref = (/ &
     (7.1428575E-02,0.5000000),  (0.1034483,0.5172414),  (0.1333333,0.5333334), &
     (0.1612903,0.5483871),      (0.1875000,0.5625000),  (0.2121212,0.5757576), &
     (0.2352941,0.5882353),      (0.2571429,0.6000000),  (0.2777778,0.6111111), &
     (0.2972973,0.6216216),      (0.3157895,0.6315789),  (0.3333333,0.6410257), &
     (0.3500000,0.6500000),      (0.3658537,0.6585366),  (0.3809524,0.6666667)  &
  /)
  complex(kind=4), dimension(ivl) :: cexp_ref = (/ &
     (1.010359,7.319654),    (-15.25874,13.06138),  (-52.28643,-15.71899), &
     (-40.83786,-142.6840),  (266.3908,-302.9699),  (1084.246,164.3603),   &
     (1216.475,2721.451),    (-4438.296,6779.491),  (-22025.60,-194.9631), &
     (-31902.92,-50666.72),  (69037.17,-147387.2),  (438521.4,-58554.19),  &
     (777987.9,917056.1),    (-955006.8,3126410.),  (-8553823.,2407299.)   &
  /)
  complex(kind=4), dimension(ivl) :: clog_ref = (/ &
     (2.649159,1.428899),  (2.727660,1.373401),  (2.802901,1.325818), &
     (2.874696,1.284745),  (2.943052,1.249046),  (3.008079,1.217806), &
     (3.069942,1.190290),  (3.128834,1.165905),  (3.184950,1.144169), &
     (3.238486,1.124691),  (3.289626,1.107149),  (3.338542,1.091277), &
     (3.385395,1.076855),  (3.430332,1.063698),  (3.473488,1.051650)  &
  /)
  complex(kind=4), dimension(ivl) :: cpowc_ref = (/ &
     (1.044926,-2.9423773E-02),  (1.043849,-3.0117445E-02),  (1.042924,-3.0676235E-02), &
     (1.042117,-3.1115681E-02),  (1.041400,-3.1451210E-02),  (1.040755,-3.1697091E-02), &
     (1.040166,-3.1866122E-02),  (1.039623,-3.1969503E-02),  (1.039117,-3.2016937E-02), &
     (1.038641,-3.2016795E-02),  (1.038190,-3.1976208E-02),  (1.037761,-3.1901326E-02), &
     (1.037349,-3.1797409E-02),  (1.036954,-3.1668913E-02),  (1.036572,-3.1519707E-02)  &
  /)
  complex(kind=4), dimension(ivl) :: cpowi_ref = (/ &
     (-4.8000002E-03,-1.4000001E-03),  (-3.9447728E-03,-1.6436552E-03),  (-3.2439446E-03,-1.7301039E-03), &
     (-2.6775934E-03,-1.7242078E-03),  (-2.2222223E-03,-1.6666667E-03),  (-1.8560381E-03,-1.5823914E-03), &
     (-1.5606422E-03,-1.4863259E-03),  (-1.3211784E-03,-1.3872373E-03),  (-1.1259147E-03,-1.2901106E-03), &
     (-9.6568046E-04,-1.1976331E-03),  (-8.3333335E-04,-1.1111111E-03),  (-7.2330894E-04,-1.0310325E-03), &
     (-6.3126005E-04,-9.5741102E-04),  (-5.5377552E-04,-8.8999636E-04),  (-4.8816571E-04,-8.2840241E-04)  &
  /)
  complex(kind=4), dimension(ivl) :: cpowk_ref = (/ &
     (2.1079999E-05,1.3440000E-05),   (1.2859633E-05,1.2967697E-05),   (7.5299176E-06,1.1224722E-05),  &
     (4.1966132E-06,9.2334549E-06),   (2.1604938E-06,7.4074073E-06),   (9.4091462E-07,5.8739570E-06),  &
     (2.2643935E-07,4.6392452E-06),   (-1.7891503E-07,3.6655763E-06),  (-3.9670152E-07,2.9051096E-06), &
     (-5.0178636E-07,2.3130619E-06),  (-5.4012344E-07,1.8518518E-06),  (-5.3985212E-07,1.4915099E-06), &
     (-5.1814663E-07,1.2087505E-06),  (-4.8542631E-07,9.8571661E-07),  (-4.4794473E-07,8.0879516E-07)  &
  /)
  complex(kind=4), dimension(ivl) :: csqrt_ref = (/ &
     (2.840962,2.463954),  (3.024653,2.479623),  (3.200970,2.499242), &
     (3.370463,2.521909),  (3.533671,2.546926),  (3.691101,2.573758), &
     (3.843219,2.601986),  (3.990446,2.631285),  (4.133164,2.661399), &
     (4.271715,2.692127),  (4.406405,2.723308),  (4.537510,2.754815), &
     (4.665278,2.786543),  (4.789932,2.818412),  (4.911671,2.850354)  &
  /)

end module complex_data_mod

module real_data_mod
  implicit none
  integer, parameter :: ivl = 31        ! 16 + 8 + 4 + 2 + 1
  real(kind=4), dimension(ivl) :: rarg1
  real(kind=4), dimension(ivl) :: rarg2
  real(kind=8), dimension(ivl) :: darg1
  real(kind=8), dimension(ivl) :: darg2

  real(kind=4), dimension(ivl) :: rabs_ref = (/ &
     1.0000000, 0.5000000, 0.3333333, 0.2500000, 0.2000000, &
     0.1666667, 0.1428571, 0.1250000, 0.1111111, 0.1000000, &
     0.0909091, 0.0833333, 0.0769231, 0.0714286, 0.0666667, &
     0.0625000, 0.0588235, 0.0555556, 0.0526316, 0.0500000, &
     0.0476190, 0.0454545, 0.0434783, 0.0416667, 0.0400000, &
     0.0384615, 0.0370370, 0.0357143, 0.0344828, 0.0333333, &
     0.0322581 &
  /)
  real(kind=4), dimension(ivl) :: racos_ref = (/ &
     0.0000000, 1.0471976, 1.2309594, 1.3181161, 1.3694384, &
     1.4033483, 1.4274487, 1.4454685, 1.4594553, 1.4706290, &
     1.4797616, 1.4873662, 1.4937972, 1.4993069, 1.5040802, &
     1.5082556, 1.5119388, 1.5152122, 1.5181404, 1.5207756, &
     1.5231593, 1.5253263, 1.5273044, 1.5291177, 1.5307857, &
     1.5323253, 1.5337509, 1.5350745, 1.5363069, 1.5374569, &
     1.5385327 &
  /)
  real(kind=4), dimension(ivl) :: rasin_ref = (/ &
     1.5707964, 0.5235988, 0.3398369, 0.2526802, 0.2013579, &
     0.1674481, 0.1433476, 0.1253278, 0.1113410, 0.1001674, &
     0.0910348, 0.0834301, 0.0769991, 0.0714895, 0.0667161, &
     0.0625408, 0.0588575, 0.0555842, 0.0526559, 0.0500209, &
     0.0476371, 0.0454702, 0.0434920, 0.0416787, 0.0400107, &
     0.0384710, 0.0370455, 0.0357219, 0.0344896, 0.0333395, &
     0.0322637 &
  /)
  real(kind=4), dimension(ivl) :: ratan_ref = (/ &
     0.7853982, 0.4636476, 0.3217506, 0.2449787, 0.1973956, &
     0.1651487, 0.1418971, 0.1243550, 0.1106572, 0.0996687, &
     0.0906599, 0.0831412, 0.0767719, 0.0713075, 0.0665682, &
     0.0624188, 0.0587558, 0.0554985, 0.0525831, 0.0499584, &
     0.0475831, 0.0454233, 0.0434509, 0.0416426, 0.0399787, &
     0.0384426, 0.0370201, 0.0356991, 0.0344691, 0.0333210, &
     0.0322469 &
  /)
  real(kind=4), dimension(ivl) :: rcos_ref = (/ &
     0.5403023, 0.8775826, 0.9449570, 0.9689124, 0.9800666, &
     0.9861432, 0.9898133, 0.9921977, 0.9938335, 0.9950042, &
     0.9958706, 0.9965298, 0.9970429, 0.9974501, 0.9977786, &
     0.9980475, 0.9982704, 0.9984572, 0.9986153, 0.9987503, &
     0.9988664, 0.9989671, 0.9990550, 0.9991321, 0.9992001, &
     0.9992604, 0.9993142, 0.9993623, 0.9994055, 0.9994445, &
     0.9994798 &
  /)
  real(kind=4), dimension(ivl) :: rsin_ref = (/ &
     0.8414710, 0.4794255, 0.3271947, 0.2474040, 0.1986693, &
     0.1658961, 0.1423717, 0.1246747, 0.1108826, 0.0998334, &
     0.0907839, 0.0832369, 0.0768472, 0.0713679, 0.0666173, &
     0.0624593, 0.0587896, 0.0555270, 0.0526073, 0.0499792, &
     0.0476011, 0.0454389, 0.0434646, 0.0416546, 0.0399893, &
     0.0384521, 0.0370286, 0.0357067, 0.0344759, 0.0333272, &
     0.0322525 &
  /)
  real(kind=4), dimension(ivl) :: rtan_ref = (/ &
     1.5574077, 0.5463025, 0.3462536, 0.2553419, 0.2027100, &
     0.1682272, 0.1438370, 0.1256551, 0.1115706, 0.1003347, &
     0.0911604, 0.0835268, 0.0770752, 0.0715503, 0.0667656, &
     0.0625815, 0.0588915, 0.0556128, 0.0526802, 0.0500417, &
     0.0476551, 0.0454859, 0.0435057, 0.0416908, 0.0400213, &
     0.0384805, 0.0370540, 0.0357295, 0.0344964, 0.0333457, &
     0.0322693 &
  /)
  !real(kind=4), dimension(ivl) :: rsincos_ref = (/ (1.0, i=1, ivl) /)
  real(kind=4), dimension(ivl) :: rcosh_ref = (/ &
     1.5430807, 1.1276259, 1.0560719, 1.0314131, 1.0200667, &
     1.0139210, 1.0102215, 1.0078226, 1.0061792, 1.0050042, &
     1.0041351, 1.0034742, 1.0029601, 1.0025522, 1.0022230, &
     1.0019537, 1.0017306, 1.0015436, 1.0013853, 1.0012503, &
     1.0011340, 1.0010332, 1.0009453, 1.0008682, 1.0008001, &
     1.0007397, 1.0006859, 1.0006378, 1.0005946, 1.0005556, &
     1.0005203 &
  /)
  real(kind=4), dimension(ivl) :: rsinh_ref = (/ &
     1.1752012, 0.5210953, 0.3395406, 0.2526123, 0.2013360, &
     0.1674394, 0.1433436, 0.1253258, 0.1113399, 0.1001668, &
     0.0910344, 0.0834298, 0.0769990, 0.0714893, 0.0667161, &
     0.0625407, 0.0588575, 0.0555841, 0.0526559, 0.0500208, &
     0.0476370, 0.0454702, 0.0434920, 0.0416787, 0.0400107, &
     0.0384710, 0.0370455, 0.0357219, 0.0344896, 0.0333395, &
     0.0322637 &
  /)
  real(kind=4), dimension(ivl) :: rtanh_ref = (/ &
     0.7615942, 0.4621172, 0.3215128, 0.2449187, 0.1973753, &
     0.1651404, 0.1418932, 0.1243530, 0.1106561, 0.0996680, &
     0.0906595, 0.0831410, 0.0767717, 0.0713073, 0.0665681, &
     0.0624187, 0.0587558, 0.0554985, 0.0525830, 0.0499584, &
     0.0475831, 0.0454233, 0.0434509, 0.0416426, 0.0399787, &
     0.0384426, 0.0370201, 0.0356991, 0.0344691, 0.0333210, &
     0.0322469 &
  /)
  real(kind=4), dimension(ivl) :: rexp_ref = (/ &
     2.7182817, 1.6487212, 1.3956125, 1.2840254, 1.2214028, &
     1.1813604, 1.1535650, 1.1331484, 1.1175190, 1.1051710, &
     1.0951694, 1.0869040, 1.0799590, 1.0740415, 1.0689391, &
     1.0644945, 1.0605880, 1.0571277, 1.0540413, 1.0512711, &
     1.0487710, 1.0465034, 1.0444373, 1.0425469, 1.0408108, &
     1.0392108, 1.0377314, 1.0363597, 1.0350841, 1.0338951, &
     1.0327840 &
  /)
  real(kind=4), dimension(ivl) :: rlog_ref = (/ &
     0.0000000, 0.6931472, 1.0986123, 1.3862944, 1.6094379, &
     1.7917595, 1.9459101, 2.0794415, 2.1972246, 2.3025851, &
     2.3978953, 2.4849067, 2.5649493, 2.6390572, 2.7080503, &
     2.7725887, 2.8332133, 2.8903718, 2.9444389, 2.9957323, &
     3.0445225, 3.0910425, 3.1354942, 3.1780539, 3.2188759, &
     3.2580965, 3.2958369, 3.3322043, 3.3672957, 3.4011974, &
     3.4339871 &
  /)
  real(kind=4), dimension(ivl) :: rlog10_ref = (/ &
     0.0000000, 0.3010300, 0.4771213, 0.6020600, 0.6989700, &
     0.7781513, 0.8450980, 0.9030900, 0.9542425, 1.0000000, &
     1.0413927, 1.0791813, 1.1139433, 1.1461281, 1.1760913, &
     1.2041200, 1.2304490, 1.2552725, 1.2787536, 1.3010300, &
     1.3222194, 1.3424227, 1.3617278, 1.3802112, 1.3979400, &
     1.4149733, 1.4313637, 1.4471580, 1.4623979, 1.4771212, &
     1.4913616 &
  /)
  real(kind=4), dimension(ivl) :: rpowc_ref = (/ &
     1.0000000, 1.5220973, 1.9083568, 2.2081790, 2.4452128, &
     2.6340396, 2.7847643, 2.9048460, 3.0000000, 3.0747149, &
     3.1325736, 3.1764700, 3.2087648, 3.2313964, 3.2459664, &
     3.2538044, 3.2560201, 3.2535396, 3.2471430, 3.2374856, &
     3.2251201, 3.2105143, 3.1940644, 3.1761060, 3.1569252, &
     3.1367643, 3.1158297, 3.0942974, 3.0723169, 3.0500152, &
     3.0275018 &
  /)
  real(kind=4), dimension(ivl) :: rpowi_ref = (/ &
         1.000,     4.000,     9.000,    16.000,    25.000, &
        36.000,    49.000,    64.000,    81.000,   100.000, &
       121.000,   144.000,   169.000,   196.000,   225.000, &
       256.000,   289.000,   324.000,   361.000,   400.000, &
       441.000,   484.000,   529.000,   576.000,   625.000, &
       676.000,   729.000,   784.000,   841.000,   900.000, &
       961.000 &
  /)
  real(kind=4), dimension(ivl) :: rpowk_ref = (/ &
            1.,       16.,       81.,      256.,      625., &
         1296.,     2401.,     4096.,     6561.,    10000., &
        14641.,    20736.,    28561.,    38416.,    50625., &
        65536.,    83521.,   104976.,   130321.,   160000., &
       194481.,   234256.,   279841.,   331776.,   390625., &
       456976.,   531441.,   614656.,   707281.,   810000., &
       923521. &
  /)
  real(kind=4), dimension(ivl) :: rsqrt_ref = (/ &
     1.0000000, 0.7071068, 0.5773503, 0.5000000, 0.4472136, &
     0.4082483, 0.3779645, 0.3535534, 0.3333333, 0.3162278, &
     0.3015113, 0.2886751, 0.2773501, 0.2672612, 0.2581989, &
     0.2500000, 0.2425356, 0.2357023, 0.2294157, 0.2236068, &
     0.2182179, 0.2132007, 0.2085144, 0.2041242, 0.2000000, &
     0.1961161, 0.1924501, 0.1889822, 0.1856953, 0.1825742, &
     0.1796053 &
  /)
  real(kind=4), dimension(ivl) :: rbessel_j0_ref = (/ &
     0.7651977E+00, 0.9384698E+00, 0.9724146E+00, 0.9844359E+00, &
     0.9900250E+00, 0.9930676E+00, 0.9949045E+00, 0.9960976E+00, &
     0.9969159E+00, 0.9975016E+00, 0.9979349E+00, 0.9982647E+00, &
     0.9985213E+00, 0.9987249E+00, 0.9988892E+00, 0.9990237E+00, &
     0.9991351E+00, 0.9992285E+00, 0.9993076E+00, 0.9993751E+00, &
     0.9994332E+00, 0.9994835E+00, 0.9995275E+00, 0.9995660E+00, &
     0.9996001E+00, 0.9996302E+00, 0.9996571E+00, 0.9996812E+00, &
     0.9997028E+00, 0.9997222E+00, 0.9997399E+00 &
  /)
  real(kind=4), dimension(ivl) :: rbessel_j1_ref = (/ &
     0.4400506E+00, 0.2422685E+00, 0.1643625E+00, 0.1240260E+00, &
     0.9950083E-01, 0.8304432E-01, 0.7124651E-01, 0.6237801E-01, &
     0.5546987E-01, 0.4993753E-01, 0.4540760E-01, 0.4163051E-01, &
     0.3843310E-01, 0.3569151E-01, 0.3331482E-01, 0.3123474E-01, &
     0.2939904E-01, 0.2776706E-01, 0.2630668E-01, 0.2499219E-01, &
     0.2380278E-01, 0.2272140E-01, 0.2173399E-01, 0.2082881E-01, &
     0.1999600E-01, 0.1922721E-01, 0.1851534E-01, 0.1785430E-01, &
     0.1723882E-01, 0.1666435E-01, 0.1612693E-01 &
  /)
  real(kind=4), dimension(ivl) :: rbessel_jn_2_ref = (/ &
     0.1149035E+00, 0.3060403E-01, 0.1376074E-01, 0.7771889E-02, &
     0.4983354E-02, 0.3464192E-02, 0.2546685E-02, 0.1950583E-02, &
     0.1541623E-02, 0.1248959E-02, 0.1032347E-02, 0.8675533E-03, &
     0.7392804E-03, 0.6374841E-03, 0.5553499E-03, 0.4881223E-03, &
     0.4324013E-03, 0.3857033E-03, 0.3461805E-03, 0.3124349E-03, &
     0.2833931E-03, 0.2582200E-03, 0.2362577E-03, 0.2169825E-03, &
     0.1999733E-03, 0.1848885E-03, 0.1714482E-03, 0.1594219E-03, &
     0.1486178E-03, 0.1388761E-03, 0.1300616E-03 &
  /)
  real(kind=4), dimension(ivl) :: rbessel_y0_ref = (/ &
     0.8825696E-01, -.4445187E+00, -.7343730E+00, -.9315730E+00, &
     -.1081105E+01, -.1201645E+01, -.1302679E+01, -.1389681E+01, &
     -.1466097E+01, -.1534239E+01, -.1595733E+01, -.1651767E+01, &
     -.1703237E+01, -.1750832E+01, -.1795098E+01, -.1836472E+01, &
     -.1875310E+01, -.1911904E+01, -.1946502E+01, -.1979311E+01, &
     -.2010506E+01, -.2040240E+01, -.2068643E+01, -.2095830E+01, &
     -.2121901E+01, -.2146943E+01, -.2171036E+01, -.2194249E+01, &
     -.2216643E+01, -.2238275E+01, -.2259195E+01 &
  /)
  real(kind=4), dimension(ivl) :: rbessel_y1_ref = (/ &
     -.7812128E+00, -.1471472E+01, -.2088166E+01, -.2704105E+01, &
     -.3323825E+01, -.3946870E+01, -.4572448E+01, -.5199936E+01, &
     -.5828878E+01, -.6458951E+01, -.7089917E+01, -.7721601E+01, &
     -.8353870E+01, -.8986625E+01, -.9619785E+01, -.1025329E+02, &
     -.1088708E+02, -.1152113E+02, -.1215540E+02, -.1278985E+02, &
     -.1342448E+02, -.1405925E+02, -.1469416E+02, -.1532918E+02, &
     -.1596431E+02, -.1659953E+02, -.1723484E+02, -.1787023E+02, &
     -.1850569E+02, -.1914121E+02, -.1977679E+02 &
  /)
  real(kind=4), dimension(ivl) :: rbessel_yn_2_ref = (/ &
     -.1650683E+01, -.5441371E+01, -.1179462E+02, -.2070127E+02, &
     -.3215715E+02, -.4616079E+02, -.6271160E+02, -.8180930E+02, &
     -.1034537E+03, -.1276448E+03, -.1543824E+03, -.1836667E+03, &
     -.2154974E+03, -.2498746E+03, -.2867984E+03, -.3262687E+03, &
     -.3682856E+03, -.4128488E+03, -.4599586E+03, -.5096149E+03, &
     -.5618176E+03, -.6165668E+03, -.6738626E+03, -.7337048E+03, &
     -.7960936E+03, -.8610286E+03, -.9285104E+03, -.9985383E+03, &
     -.1071113E+04, -.1146234E+04, -.1223902E+04 &
  /)

  real(kind=4), dimension(ivl) :: rgamma_ref = (/ &
      1.0000000,     1.7724538,     2.6789386,     3.6256099, &
      4.5908437,     5.5663161,     6.5480628,     7.5339427, &
      8.5226889,     9.5135088,    10.5058756,    11.4994268, &
     12.4939108,    13.4891329,    14.4849615,    15.4812813, &
     16.4780140,    17.4750900,    18.4724598,    19.4700851, &
     20.4679222,    21.4659557,    22.4641476,    23.4624882, &
     24.4609566,    25.4595356,    26.4582176,    27.4569855, &
     28.4558468,    29.4547787,    30.4537735 &
  /)
  real(kind=4), dimension(ivl) :: rhypot_ref = (/ &
     2.2360680, 1.1180340, 0.7453560, 0.5590170, 0.4472136, &
     0.3726780, 0.3194383, 0.2795085, 0.2484520, 0.2236068, &
     0.2032789, 0.1863390, 0.1720052, 0.1597192, 0.1490712, &
     0.1397543, 0.1315334, 0.1242260, 0.1176878, 0.1118034, &
     0.1064794, 0.1016395, 0.0972203, 0.0931695, 0.0894427, &
     0.0860026, 0.0828173, 0.0798596, 0.0771058, 0.0745356, &
     0.0721312 &
  /)
  real(kind=4), dimension(ivl) :: rerf_ref = (/ &
     0.8427008, 0.5204999, 0.3626481, 0.2763264, 0.2227026, &
     0.1863363, 0.1601071, 0.1403162, 0.1248614, 0.1124629, &
     0.1022980, 0.0938144, 0.0866275, 0.0804617, 0.0751140, &
     0.0704320, 0.0662988, 0.0626233, 0.0593336, 0.0563720, &
     0.0536918, 0.0512547, 0.0490291, 0.0469886, 0.0451111, &
     0.0433778, 0.0417727, 0.0402821, 0.0388942, 0.0375987, &
     0.0363867 &
  /)
  real(kind=4), dimension(ivl) :: rerfc_ref = (/ &
     0.1572992, 0.4795001, 0.6373519, 0.7236736, 0.7772974, &
     0.8136637, 0.8398929, 0.8596838, 0.8751386, 0.8875371, &
     0.8977020, 0.9061856, 0.9133725, 0.9195384, 0.9248860, &
     0.9295681, 0.9337012, 0.9373767, 0.9406664, 0.9436280, &
     0.9463083, 0.9487454, 0.9509709, 0.9530114, 0.9548889, &
     0.9566222, 0.9582273, 0.9597179, 0.9611058, 0.9624013, &
     0.9636133 &
  /)
  real(kind=4), dimension(ivl) :: rerfc_scaled_ref = (/ &
     0.4275836, 0.6156903, 0.7122529, 0.7703465, 0.8090195, &
     0.8365823, 0.8572096, 0.8732218, 0.8860098, 0.8964570, &
     0.9051518, 0.9125005, 0.9187931, 0.9242418, 0.9290058, &
     0.9332062, 0.9369376, 0.9402743, 0.9432757, 0.9459901, &
     0.9484565, 0.9507076, 0.9527703, 0.9546673, 0.9564180, &
     0.9580383, 0.9595426, 0.9609428, 0.9622492, 0.9634712, &
     0.9646165 &
  /)
end module real_data_mod

module pass_fail_mod
  integer :: ntest = 0
  integer :: nfail = 0
end module pass_fail_mod

module complex_diff_mod
  implicit none

  ! Relative tolerance
  real(kind=4), parameter :: reltol = 1.0e-4 ! Works for X86-64 and POWER
  real(kind=4), parameter :: abstol = 0.0    ! 1.0e-7
  contains

  logical function isclose(a, b, rtol_opt, atol_opt) result(tf)
  !
  ! isclose(a, b, relative_tolerance, absolute_tolerance)
  ! Algorithm suspiciously close to Python's cmath.isclose().
  !
  complex(kind=4), intent(in) :: a
  complex(kind=4), intent(in) :: b
  real(kind=4), intent(in), optional :: rtol_opt
  real(kind=4), intent(in), optional :: atol_opt
  real(kind=4) :: rtol
  real(kind=4) :: atol
  real(kind=4) :: r_diff
  rtol = reltol
  atol = abstol
  if (present(rtol_opt)) rtol = rtol_opt
  if (present(atol_opt)) atol = atol_opt

  r_diff = abs(a - b)
  tf = .true.
  if (a /= b) then
    tf = ((( r_diff < rtol * abs(b)) .or. &
           ( r_diff < rtol * abs(a))) .or. &
           ( r_diff <= atol))
  end if
  end function isclose

end module complex_diff_mod

module real_diff_mod
  implicit none

  ! Relative tolerance
  real(kind=4), parameter :: reltol = 1.0e-5 ! Works for X86-64 and POWER
  real(kind=4), parameter :: abstol = 0.0    ! 1.0e-7
  contains

  logical function isclose(a, b, rtol_opt, atol_opt) result(tf)
  !
  ! isclose(a, b, relative_tolerance, absolute_tolerance)
  ! Algorithm suspiciously close to Python's cmath.isclose().
  !
  real(kind=4), intent(in) :: a
  real(kind=4), intent(in) :: b
  real(kind=4), intent(in), optional :: rtol_opt
  real(kind=4), intent(in), optional :: atol_opt
  real(kind=4) :: rtol
  real(kind=4) :: atol
  real(kind=4) :: r_diff
  rtol = reltol
  atol = abstol
  if (present(rtol_opt)) rtol = rtol_opt
  if (present(atol_opt)) atol = atol_opt

  r_diff = abs(a - b)
  tf = .true.
  if (a /= b) then
    tf = ((( r_diff < rtol * abs(b)) .or. &
           ( r_diff < rtol * abs(a))) .or. &
           ( r_diff <= atol))
  end if
  end function isclose

  logical function fcmp(a, b, r_opt) result(tf)
  ! Floating compare per D.E. Knuth in Section 4.2.2 of Seminumerical Algorithms
  ! As coded not suitable for testing around zeros.
  ! fcmp - currently not used in this test.
  implicit none
  real(kind=4), intent(in) :: a         ! 1st argument 
  real(kind=4), intent(in) :: b         ! 2nd argument
  real(kind=4), intent(in), optional :: r_opt         ! relative difference

  real(kind=4) :: r                     ! local - relative difference
  real(kind=4) :: d                     ! diff
  real(kind=4) :: sr                    ! relative difference scaled

  integer :: e                          ! exponent

  r = reltol
  if (present(r_opt)) r = r_opt
  e = exponent(max(abs(a), abs(b)))
  sr = scale(r, e)
  d = a - b

  tf = d < sr .and. d > -sr
  end function fcmp
end module real_diff_mod

subroutine zcompare_all(z, cref, n, tag)
  use pass_fail_mod
  use complex_diff_mod
  implicit none
  integer, intent(in) :: n
  complex(kind=8), dimension(n), intent(in) :: z
  complex(kind=4), dimension(n), intent(in) :: cref
  character(len=*) :: tag
  complex(kind=4) :: z_4
  integer :: i
  logical :: good

  good = .true.
  ntest = ntest + 1
  do i = 1, n
    z_4 = z(i)
    if (.not. isclose(z_4, cref(i))) then
      if (good) then
        print*, 'bad :    ', tag
      end if
      good = .false.
      print*, 'Element(s)', tag, i, z(i), cref(i)
      !print '(2(1x,z16))', transfer(z(i), 1_8), transfer(zref(i), 1_8)
    end if
  end do
  if (good) then
    print*, 'good:    ', tag
  else
   nfail = nfail + 1
  end if
end subroutine zcompare_all

subroutine ccompare_all(c, cref, n, tag)
  use pass_fail_mod
  use complex_diff_mod
  implicit none
  integer, intent(in) :: n
  complex(kind=4), dimension(n), intent(in) :: c
  complex(kind=4), dimension(n), intent(in) :: cref
  character(len=*) :: tag
  integer :: i
  logical :: good

  good = .true.
  ntest = ntest + 1
  do i = 1, n
    if (.not. isclose(c(i), cref(i))) then
      if (good) then
        print*, 'bad :    ', tag
      end if
      good = .false.
      print*, 'Element(s)', tag, i, c(i), cref(i)
      !print '(2(1x,z8))', transfer(c(i), 1_4), transfer(cref(i), 1_4)
    end if
  end do
  if (good) then
    print*, 'good:    ', tag
  else
   nfail = nfail + 1
  end if
end subroutine ccompare_all

subroutine do_complex
  use complex_data_mod
  use pass_fail_mod
  implicit none
  complex(kind=4), dimension(ivl) :: cval
  complex(kind=8), dimension(ivl) :: zval
  integer :: i


  carg1 = (/ (cmplx(+1.0+(0*ivl)+i, -2.0+(1*ivl)+i), i = 1, ivl) /)
  carg2 = (/ (cmplx(-3.0+(2*ivl)+i, +4.0+(3*ivl)+i), i = 1, ivl) /)
  zarg1 = carg1
  zarg2 = carg2

! Inline
  cval = cmplx(abs(carg1), 0)
  call ccompare_all(cval, cabs_ref, ivl, 'cabs')
!  __mth_i_cdabs
  zval = cmplx(abs(zarg1), 0)
  call zcompare_all(zval, cabs_ref, ivl, 'cdabs')

!  __mth_i_cacos
  cval = acos(carg1)
  call ccompare_all(cval, cacos_ref, ivl, 'cacos')
!  __mth_i_cdacos
  zval = acos(zarg1)
  call zcompare_all(zval, cacos_ref, ivl, 'cdacos')

!  __mth_i_casin
  cval = asin(carg1)
  call ccompare_all(cval, casin_ref, ivl, 'casin')
!  __mth_i_cdasin
  zval = asin(zarg1)
  call zcompare_all(zval, casin_ref, ivl, 'cdasin')

!  __mth_i_catan
  cval = atan(carg1)
  call ccompare_all(cval, catan_ref, ivl, 'catan')
!  __mth_i_cdatan
  zval = atan(zarg1)
  call zcompare_all(zval, catan_ref, ivl, 'cdatan')

!  __mth_i_ccos
  cval = cos(carg1)
  call ccompare_all(cval, ccos_ref, ivl, 'ccos')
!  __mth_i_cdcos
  zval = cos(zarg1)
  call zcompare_all(zval, ccos_ref, ivl, 'cdcos')

!  __mth_i_csin
  cval = sin(carg1)
  call ccompare_all(cval, csin_ref, ivl, 'csin')
!  __mth_i_cdsin
  zval = sin(zarg1)
  call zcompare_all(zval, csin_ref, ivl, 'cdsin')

!  __mth_i_ctan
  cval = tan(carg1)
  call ccompare_all(cval, ctan_ref, ivl, 'ctan')
!  __mth_i_cdtan
  zval = tan(zarg1)
  call zcompare_all(zval, ctan_ref, ivl, 'cdtan')

!  __mth_i_ccosh
  cval = cosh(carg1)
  call ccompare_all(cval, ccosh_ref, ivl, 'ccosh')
!  __mth_i_cdcosh
  zval = cosh(zarg1)
  call zcompare_all(zval, ccosh_ref, ivl, 'cdcosh')

!  __mth_i_csinh
  cval = sinh(carg1)
  call ccompare_all(cval, csinh_ref, ivl, 'csinh')
!  __mth_i_cdsinh
  zval = sinh(zarg1)
  call zcompare_all(zval, csinh_ref, ivl, 'cdsinh')

!  __mth_i_ctanh
  cval = tanh(carg1)
  call ccompare_all(cval, ctanh_ref, ivl, 'ctanh')
!  __mth_i_cdtanh
  zval = tanh(zarg1)
  call zcompare_all(zval, ctanh_ref, ivl, 'cdtanh')

!no-support-yet  !  __mth_i_cacosh
!no-support-yet    cval = acosh(ccosh_ref)
!no-support-yet    call ccompare_all(cval, carg1, ivl, 'cacosh')
!no-support-yet  !  __mth_i_cdacosh
!no-support-yet    zval = acosh(cmplx(ccosh_ref,kind=8))
!no-support-yet    call zcompare_all(zval, carg1, ivl, 'cdacosh')
!no-support-yet  
!no-support-yet  !  __mth_i_casinh
!no-support-yet    cval = asinh(csinh_ref)
!no-support-yet    call ccompare_all(cval, carg1, ivl, 'casinh')
!no-support-yet  !  __mth_i_cdasinh
!no-support-yet    zval = asinh(csinh_ref)
!no-support-yet    call zcompare_all(zval, carg1, ivl, 'cdasinh')
!no-support-yet  
!no-support-yet  !  __mth_i_catanh
!no-support-yet    cval = atanh(ctanh_ref)
!no-support-yet    call ccompare_all(cval, carg1, ivl, 'catanh')
!no-support-yet  !  __mth_i_cdatanh
!no-support-yet    zval = atanh(ctanh_ref)
!no-support-yet    call zcompare_all(zval, carg1, ivl, 'cdatanh')


! X86-64: __fsc_div_vex
! POWER:  __mth_i_cdiv
  cval = carg1 / carg2
  call ccompare_all(cval, cdiv_ref, ivl, 'cdiv')
! X86-64: __fsz_div_vex
  zval = zarg1 / zarg2
  call zcompare_all(zval, cdiv_ref, ivl, 'cddiv')

! X86-64: inline
! POWER:  __mth_i_cdiv
  cval = carg1 / real(carg2)
  call ccompare_all(cval, cdivr_ref, ivl, 'cdivr')
! X86-64 inline
! POWER:  __mth_i_cddiv
  zval = zarg1 / real(zarg2)
  call zcompare_all(zval, cdivr_ref, ivl, 'cddivd')

! __mth_i_cexp
! POWER:  __mth_i_cexp
  cval = exp(carg1)
  call ccompare_all(cval, cexp_ref, ivl, 'cexp')
! X86-64: __fsz_exp_vex
! POWER:  __mth_i_cdexp
  zval = exp(zarg1)
  call zcompare_all(zval, cexp_ref, ivl, 'cdexp')

!  __mth_i_clog
  cval = log(carg1)
  call ccompare_all(cval, clog_ref, ivl, 'clog')
!  __mth_i_cdlog
  zval = log(zarg1)
  call zcompare_all(zval, clog_ref, ivl, 'cdlog')

!  __mth_i_cpowc
  cval = carg1 ** (1.0 / carg2)
  call ccompare_all(cval, cpowc_ref, ivl, 'cpowc')
!  __mth_i_cdpowcd
  zval = zarg1 ** (1.0 / zarg2)
  call zcompare_all(zval, cpowc_ref, ivl, 'cdpowcd')

!  __mth_i_cpowi
  cval = carg1 ** -2_4
  call ccompare_all(cval, cpowi_ref, ivl, 'cpowi')
!  __mth_i_cdpowi
  zval = zarg1 ** -2_4
  call zcompare_all(zval, cpowi_ref, ivl, 'cdpowi')

!  __mth_i_cpowk
  cval = carg1 ** -4_8
  call ccompare_all(cval, cpowk_ref, ivl, 'cpowk')
!  __mth_i_cdpowk
  zval = zarg1 ** -4_8
  call zcompare_all(zval, cpowk_ref, ivl, 'cdpowk')

!  __mth_i_csqrt
  cval = sqrt(carg1)
  call ccompare_all(cval, csqrt_ref, ivl, 'csqrt')
!  __mth_i_cdsqrt
  zval = sqrt(zarg1)
  call zcompare_all(zval, csqrt_ref, ivl, 'cdsqrt')


end subroutine do_complex

subroutine dcompare_all(d, rref, n, tag)
  use pass_fail_mod
  use real_diff_mod
  implicit none
  integer, intent(in) :: n
  real(kind=8), dimension(n), intent(in) :: d
  real(kind=4), dimension(n), intent(in) :: rref
  character(len=*) :: tag
  real(kind=4) :: d_4
  integer :: i
  logical :: good

  good = .true.
  ntest = ntest + 1
  do i = 1, n
    d_4 = d(i)
    if (.not. isclose(d_4, rref(i))) then
      if (good) then
        print*, 'bad :    ', tag
      end if
      good = .false.
      print*, 'Element(s)', tag, i, d(i), rref(i)
      !print '(2(1x,z16))', transfer(z(i), 1_8), transfer(zref(i), 1_8)
    end if
  end do
  if (good) then
    print*, 'good:    ', tag
  else
   nfail = nfail + 1
  end if
end subroutine dcompare_all

subroutine rcompare_all(r, rref, n, tag)
  use pass_fail_mod
  use real_diff_mod
  implicit none
  integer, intent(in) :: n
  real(kind=4), dimension(n), intent(in) :: r
  real(kind=4), dimension(n), intent(in) :: rref
  character(len=*) :: tag
  integer :: i
  logical :: good

  good = .true.
  ntest = ntest + 1
  do i = 1, n
    !if (fcmp(r(i), rref(i)) /= isclose(r(i), rref(i))) then
    !  print*, r(i), rref(i), fcmp(r(i), rref(i)), isclose(r(i), rref(i))
    !  stop
    !end if
    if (.not. isclose(r(i), rref(i))) then
      if (good) then
        print*, 'bad :    ', tag
      end if
      good = .false.
      print*, 'Element(s)', tag, i, r(i), rref(i)
      !print '(2(1x,z8))', transfer(c(i), 1_4), transfer(cref(i), 1_4)
    end if
  end do
  if (good) then
    print*, 'good:    ', tag
  else
   nfail = nfail + 1
  end if
end subroutine rcompare_all

subroutine do_real
  use real_data_mod
  use pass_fail_mod
  implicit none
  real(kind=4), dimension(ivl) :: rval
  real(kind=8), dimension(ivl) :: dval
  integer :: i


  rarg1 = (/ (1.0 / ((0*ivl)+i), i = 1, ivl) /)
  rarg2 = (/ (1.0 / ((1*ivl)+i), i = 1, ivl) /)
  darg1 = rarg1
  darg2 = rarg2

! Inline
  rval = abs(-rarg1)
  call rcompare_all(rval, rabs_ref, ivl, 'abs')
!  __mth_i_dabs
  dval = abs(-darg1)
  call dcompare_all(dval, rabs_ref, ivl, 'dabs')


!  __mth_i_acos
  rval = acos(rarg1)
  call rcompare_all(rval, racos_ref, ivl, 'acos')
!  __mth_i_dacos
  dval = acos(darg1)
  call dcompare_all(dval, racos_ref, ivl, 'dacos')

!  __mth_i_asin
  rval = asin(rarg1)
  call rcompare_all(rval, rasin_ref, ivl, 'asin')
!  __mth_i_dasin
  dval = asin(darg1)
  call dcompare_all(dval, rasin_ref, ivl, 'dasin')

!  __mth_i_atan
  rval = atan(rarg1)
  call rcompare_all(rval, ratan_ref, ivl, 'atan')
!  __mth_i_datan
  dval = atan(darg1)
  call dcompare_all(dval, ratan_ref, ivl, 'datan')

!  __mth_i_cos
  rval = cos(rarg1)
  call rcompare_all(rval, rcos_ref, ivl, 'cos')
!  __mth_i_dcos
  dval = cos(darg1)
  call dcompare_all(dval, rcos_ref, ivl, 'dcos')

!  __mth_i_sin
  rval = sin(rarg1)
  call rcompare_all(rval, rsin_ref, ivl, 'sin')
!  __mth_i_dsin
  dval = sin(darg1)
  call dcompare_all(dval, rsin_ref, ivl, 'dsin')

!  __mth_i_tan
  rval = tan(rarg1)
  call rcompare_all(rval, rtan_ref, ivl, 'tan')
!  __mth_i_dtan
  dval = tan(darg1)
  call dcompare_all(dval, rtan_ref, ivl, 'dtan')

!  __mth_i_sincos
  rval = sin(rarg1)**2 + cos(rarg1)**2
  call rcompare_all(rval, (/ (1.0, i=1, ivl) /), ivl, 'sincos')
!  __mth_i_dsincos
  dval = sin(darg1)**2 + cos(darg1)**2
  call rcompare_all(rval, (/ (1.0, i=1, ivl) /), ivl, 'dsincos')

!  __mth_i_cosh
  rval = cosh(rarg1)
  call rcompare_all(rval, rcosh_ref, ivl, 'cosh')
!  __mth_i_dcosh
  dval = cosh(darg1)
  call dcompare_all(dval, rcosh_ref, ivl, 'dcosh')

!  __mth_i_sinh
  rval = sinh(rarg1)
  call rcompare_all(rval, rsinh_ref, ivl, 'sinh')
!  __mth_i_dsinh
  dval = sinh(darg1)
  call dcompare_all(dval, rsinh_ref, ivl, 'dsinh')

!  __mth_i_tanh
  rval = tanh(rarg1)
  call rcompare_all(rval, rtanh_ref, ivl, 'tanh')
!  __mth_i_dtanh
  dval = tanh(darg1)
  call dcompare_all(dval, rtanh_ref, ivl, 'dtanh')

  if (.false.) then
!  __mth_i_acosh
  rval = acosh(rcosh_ref)
  call rcompare_all(rval, rarg1, ivl, 'acosh')
!  __mth_i_dacosh
  dval = acosh(real(rcosh_ref,kind=8))
  call dcompare_all(dval, rarg1, ivl, 'dacosh')
  endif

!  __mth_i_asinh
  rval = asinh(rsinh_ref)
  call rcompare_all(rval, rarg1, ivl, 'asinh')
!  __mth_i_dasinh
  dval = asinh(real(rsinh_ref,kind=8))
  call dcompare_all(dval, rarg1, ivl, 'dasinh')

!  __mth_i_atanh
  rval = atanh(rtanh_ref)
  call rcompare_all(rval, rarg1, ivl, 'atanh')
!  __mth_i_datanh
  dval = atanh(real(rtanh_ref,kind=8))
  call dcompare_all(dval, rarg1, ivl, 'datanh')

! __mth_i_exp
! POWER:  __mth_i_exp
  rval = exp(rarg1)
  call rcompare_all(rval, rexp_ref, ivl, 'exp')
! X86-64: __fsz_exp_vex
! POWER:  __mth_i_dexp
  dval = exp(darg1)
  call dcompare_all(dval, rexp_ref, ivl, 'dexp')

!  __mth_i_log
  rval = log(1.0 / rarg1)
  call rcompare_all(rval, rlog_ref, ivl, 'log')
!  __mth_i_dlog
  dval = log(1.0 / darg1)
  call dcompare_all(dval, rlog_ref, ivl, 'dlog')

!  __mth_i_log10
  rval = log10(1.0 / rarg1)
  call rcompare_all(rval, rlog10_ref, ivl, 'log10')
!  __mth_i_dlog10
  dval = log10(1.0 / darg1)
  call dcompare_all(dval, rlog10_ref, ivl, 'dlog10')

!  __mth_i_powc
  rval = (1.0 / rarg1) ** (20. * rarg2)
  call rcompare_all(rval, rpowc_ref, ivl, 'pow')
!  __mth_i_dpowcd
  dval = (1.0 / darg1) ** (20. * darg2)
  call dcompare_all(dval, rpowc_ref, ivl, 'dpowd')

!  __mth_i_powi
  rval = rarg1 ** -2_4
  call rcompare_all(rval, rpowi_ref, ivl, 'powi')
!  __mth_i_dpowi
  dval = darg1 ** -2_4
  call dcompare_all(dval, rpowi_ref, ivl, 'dpowi')

!  __mth_i_powk
  rval = rarg1 ** -4_8
  call rcompare_all(rval, rpowk_ref, ivl, 'powk')
!  __mth_i_dpowk
  dval = darg1 ** -4_8
  call dcompare_all(dval, rpowk_ref, ivl, 'dpowk')

!  __mth_i_sqrt
  rval = sqrt(rarg1)
  call rcompare_all(rval, rsqrt_ref, ivl, 'sqrt')
!  __mth_i_dsqrt
  dval = sqrt(darg1)
  call dcompare_all(dval, rsqrt_ref, ivl, 'dsqrt')

  rval = bessel_j0(rarg1)
  call rcompare_all(rval, rbessel_j0_ref, ivl, 'bessel_j0')
  dval = bessel_j0(darg1)
  call dcompare_all(dval, rbessel_j0_ref, ivl, 'dbessel_j0')

  rval = bessel_j1(rarg1)
  call rcompare_all(rval, rbessel_j1_ref, ivl, 'bessel_j1')
  dval = bessel_j1(darg1)
  call dcompare_all(dval, rbessel_j1_ref, ivl, 'dbessel_j1')

  rval = bessel_jn(2, rarg1)
  call rcompare_all(rval, rbessel_jn_2_ref, ivl, 'bessel_jn(2)')
  dval = bessel_jn(2, darg1)
  call dcompare_all(dval, rbessel_jn_2_ref, ivl, 'dbessel_jn(2)')

  rval = bessel_y0(rarg1)
  call rcompare_all(rval, rbessel_y0_ref, ivl, 'bessel_y0')
  dval = bessel_y0(darg1)
  call dcompare_all(dval, rbessel_y0_ref, ivl, 'dbessel_y0')

  rval = bessel_y1(rarg1)
  call rcompare_all(rval, rbessel_y1_ref, ivl, 'bessel_y1')
  dval = bessel_y1(darg1)
  call dcompare_all(dval, rbessel_y1_ref, ivl, 'dbessel_y1')

  rval = bessel_yn(2, rarg1)
  call rcompare_all(rval, rbessel_yn_2_ref, ivl, 'bessel_yn(2)')
  dval = bessel_yn(2, darg1)
  call dcompare_all(dval, rbessel_yn_2_ref, ivl, 'dbessel_yn(2)')

  rval = gamma(rarg1)
  call rcompare_all(rval, rgamma_ref, ivl, 'gamma')
  dval = gamma(darg1)
  call dcompare_all(dval, rgamma_ref, ivl, 'dgamma')

  rval = hypot(rarg1, 2*rarg1)
  call rcompare_all(rval, rhypot_ref, ivl, 'hypot')
  dval = hypot(darg1, 2*rarg1)
  call dcompare_all(dval, rhypot_ref, ivl, 'dhypot')

  rval = erf(rarg1)
  call rcompare_all(rval, rerf_ref, ivl, 'erf')
  dval = erf(darg1)
  call dcompare_all(dval, rerf_ref, ivl, 'derf')

  rval = erfc(rarg1)
  call rcompare_all(rval, rerfc_ref, ivl, 'erfc')
  dval = erfc(darg1)
  call dcompare_all(dval, rerfc_ref, ivl, 'derfc')

  rval = erfc_scaled(rarg1)
  call rcompare_all(rval, rerfc_scaled_ref, ivl, 'erfc_scaled')
  dval = erfc_scaled(darg1)
  call dcompare_all(dval, rerfc_scaled_ref, ivl, 'derfc_scaled')

! Format statements used to build some of the static reference results.
!  print '(5(1x, f9.7, ","), " &")', rval ; stop
!  print '(4(1x, e13.7, ","), " &")', rval ; stop
end subroutine do_real

program test
  use pass_fail_mod
  implicit none
  call do_complex
  call do_real
  if (nfail == 0) then
    print*, " --- ", ntest, "tests PASSED. 0 tests failed."
  else
    print*, " --- ", ntest-nfail, "tests passed.",  nfail, "tests FAILED."
  end if
end program test

subroutine zprint(z)
  complex(kind=8), intent(in) :: z
  !print '(2(1x, z16))', transfer(real(z,8), 1_8), transfer(aimag(z), 1_8)
end subroutine zprint
