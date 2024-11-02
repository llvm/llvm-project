! RUN: %python %S/test_folding.py %s %flang_fc1

! Check intrinsic function folding with host runtime library

module m
  real(2), parameter :: eps2 = 0.001_2
  real(2), parameter :: eps3 = 0.001_3
  real(4), parameter :: eps4 = 0.000001_4
  real(8), parameter :: eps8 = 0.000000000000001_8

  ! These eps have been set so that constant folding of intrinsic
  ! functions that use host runtime can be tested independently of
  ! the underlying math library used.
  ! C++ <cmath> and libpgmath precise, relaxed and fast libraries pass the test.
  ! It may have to be relaxed to pass on all architectures.
  ! The purpose is to check that the "correct" runtime functions are mapped
  ! to intrinsic functions but not to test the stability between different libraries.
  ! The eps should just be low enough to give confidence that intrinsic
  ! functions are mapped to runtime functions implementing the same math
  ! function.
  ! Compared values were selected to be around 1 +/- 0.5 so that eps is meaningful.
  ! Expected values come from libpgmath-precise for Real(4) and Real(8) and
  ! were computed on X86_64.

  logical, parameter :: test_sign_i4 = sign(1_4,2) == 1_4 .and. sign(1_4,-3_8) == -1_4
  logical, parameter :: test_sign_i8 = sign(1_8,2) == 1_8 .and. sign(1_8,-3_8) == -1_8

! Real scalar intrinsic function tests
#define TEST_FLOATING(name, result, expected, t, k) \
  t(kind = k), parameter ::res_##name##_##t##k = result; \
  t(kind = k), parameter ::exp_##name##_##t##k = expected; \
  logical, parameter ::test_##name##_##t##k = abs(res_##name##_##t##k - exp_##name##_##t##k).LE.(eps##k)

#define TEST_R2(name, result, expected) TEST_FLOATING(name, result, expected, real, 2)
#define TEST_R3(name, result, expected) TEST_FLOATING(name, result, expected, real, 3)
#define TEST_R4(name, result, expected) TEST_FLOATING(name, result, expected, real, 4)
#define TEST_R8(name, result, expected) TEST_FLOATING(name, result, expected, real, 8)
#define TEST_C4(name, result, expected) TEST_FLOATING(name, result, expected, complex, 4)
#define TEST_C8(name, result, expected) TEST_FLOATING(name, result, expected, complex, 8)

! REAL(4) tests.

  logical, parameter :: test_abs_r4 = abs(-2._4).EQ.(2._4)
  TEST_R4(acos, acos(0.5_4), 1.0471975803375244140625_4)
  TEST_R4(acosh, acosh(1.5_4), 0.96242368221282958984375_4)
  logical, parameter :: test_aint1 = aint(2.783).EQ.(2.)
  logical, parameter :: test_anint1 = anint(2.783).EQ.(3.)
  logical, parameter :: test_floor1 = floor(-2.783).EQ.(-3.)
  logical, parameter :: test_floor2 = floor(2.783).EQ.(2.)
  logical, parameter :: test_ceiling1 = ceiling(-2.783).EQ.(-2.)
  logical, parameter :: test_ceiling2 = ceiling(2.783).EQ.(3.)
  TEST_R4(asin, asin(0.9_4), 1.11976945400238037109375_4)
  TEST_R4(asinh, asinh(1._4), 0.881373584270477294921875_4)
  TEST_R4(atan, atan(1.5_4), 0.982793748378753662109375_4)
  TEST_R4(atan2, atan2(1.5_4, 1._4), 0.982793748378753662109375_4)
  TEST_R4(atan_2, atan(1.5_4, 1._4), 0.982793748378753662109375_4)
  TEST_R4(atanh, atanh(0.8_4), 1.098612308502197265625_4)
  TEST_R4(cos, cos(0.5_4), 0.877582550048828125_4)
  TEST_R4(cosh, cosh(0.1_4), 1.0050041675567626953125_4)
  TEST_R4(erf, erf(1._4), 0.842700779438018798828125_4)
  TEST_R4(erfc, erfc(0.1_4), 0.887537062168121337890625_4)
  TEST_R4(exp, exp(0.1_4), 1.1051709651947021484375_4)
  TEST_R4(gamma, gamma(0.9_4), 1.06862866878509521484375_4)
  TEST_R4(hypot, hypot(1.1_4, 0.1_4), 1.10453617572784423828125_4)
  TEST_R4(log, log(3._4), 1.098612308502197265625_4)
  TEST_R4(log10, log10(10.5_4), 1.02118933200836181640625_4)
  TEST_R4(log_gamma, log_gamma(3.5_4), 1.20097362995147705078125_4)
  TEST_R4(mod, mod(-8.1_4, 5._4), (-3.1000003814697265625_4))
  TEST_R4(real, real(z'3f800000'), 1._4)
  logical, parameter :: test_sign_r4 = sign(1._4,2._8) == 1._4 .and. sign(1._4,-2._4) == -1._4
  TEST_R4(sin, sin(1.6_4), 0.99957358837127685546875_4)
  TEST_R4(sinh, sinh(0.9_4), 1.0265166759490966796875_4)
  TEST_R4(sqrt, sqrt(1.1_4), 1.0488088130950927734375_4)
  TEST_R4(tan, tan(0.8_4), 1.0296385288238525390625_4)
  TEST_R4(tanh, tanh(3._4), 0.995054781436920166015625_4)

! REAL(8) tests.

  logical, parameter :: test_abs_r8 = abs(-2._8).EQ.(2._8)
  TEST_R8(acos, acos(0.5_8), &
    1.047197551196597853362391106202267110347747802734375_8)
  TEST_R8(acosh, acosh(1.5_8), &
    0.9624236501192069415111518537742085754871368408203125_8)
  TEST_R8(asin, asin(0.9_8), &
    1.119769514998634196700777465594001114368438720703125_8)
  TEST_R8(asinh, asinh(1._8), &
    0.88137358701954304773806825323845259845256805419921875_8)
  TEST_R8(atan, atan(1.5_8), &
    0.98279372324732905408239957978366874158382415771484375_8)
  TEST_R8(atan2, atan2(1.5_8, 1._8), &
    0.98279372324732905408239957978366874158382415771484375_8)
  TEST_R8(atan_2, atan(1.5_8, 1._8), &
    0.98279372324732905408239957978366874158382415771484375_8)
  TEST_R8(atanh, atanh(0.8_8), &
    1.0986122886681097821082175869378261268138885498046875_8)
  TEST_R8(cos, cos(0.5_8), &
    0.8775825618903727587394314468838274478912353515625_8)
  TEST_R8(cosh, cosh(0.1_8), &
    1.0050041680558035039894093642942607402801513671875_8)
  TEST_R8(erf, erf(1._8), &
    0.84270079294971489414223242420121096074581146240234375_8)
  TEST_R8(erfc, erfc(0.1_8), &
    0.8875370839817151580319887216319330036640167236328125_8)
  TEST_R8(exp, exp(0.1_8), &
    1.10517091807564771244187795673497021198272705078125_8)
  TEST_R8(gamma, gamma(0.9_8), &
    1.0686287021193192625645451698801480233669281005859375_8)
  TEST_R8(hypot, hypot(1.1_8, 0.1_8), &
    1.1045361017187260710414875575224868953227996826171875_8)
  TEST_R8(log, log(3._8), &
    1.0986122886681097821082175869378261268138885498046875_8)
  TEST_R8(log10, log10(10.5_8), &
    1.0211892990699380501240511875948868691921234130859375_8)
  TEST_R8(log_gamma, log_gamma(3.5_8), &
    1.200973602347074287166606154642067849636077880859375_8)
  TEST_R8(mod, mod(-8.1_8, 5._8), &
    (-3.0999999999999996447286321199499070644378662109375_8))
  TEST_R8(real, real(z'3ff0000000000000',8), 1._8)
  logical, parameter :: test_sign_r8 = sign(1._8,2._8) == 1._8 .and. sign(1._8,-2._4) == -1._8
  TEST_R8(sin, sin(1.6_8), &
    0.99957360304150510987852840116829611361026763916015625_8)
  TEST_R8(sinh, sinh(0.9_8), &
    1.0265167257081753149350333842448890209197998046875_8)
  TEST_R8(sqrt, sqrt(1.1_8), &
    1.048808848170151630796453900984488427639007568359375_8)
  TEST_R8(tan, tan(0.8_8), &
    1.0296385570503641115891468871268443763256072998046875_8)
  TEST_R8(tanh, tanh(3._8), &
    0.995054753686730464323773048818111419677734375_8)

! COMPLEX(4) tests.

  logical, parameter :: test_abs_c4 = abs(abs((1.1_4, 0.1_4)) &
    - 1.10453617572784423828125_4).LE.(eps4)
  TEST_C4(acos, acos((0.7_4, 1.1_4)), &
    (1.11259567737579345703125_4, -1.03283786773681640625_4))
  TEST_C4(acosh, acosh((0.7_4, 1.1_4)), &
    (1.03283774852752685546875_4, 1.11259555816650390625_4))
  TEST_C4(asin, asin((1.4_4, 0.7_4)), &
    (1.0101039409637451171875_4,1.08838176727294921875_4))
  TEST_C4(asinh, asinh((0.7_4, 1.4_4)), &
    (1.08838176727294921875_4,1.0101039409637451171875_4))
  TEST_C4(atan, atan((0.2_4, 1.1_4)), &
    (1.06469786167144775390625_4,1.12215900421142578125_4))
  TEST_C4(atanh, atanh((1.1_4, 0.2_4)), &
    (1.12215900421142578125_4,1.06469786167144775390625_4))
  TEST_C4(cmplx, cmplx(z'bf800000',z'3f000000'), (-1._4,0.5_4))
  TEST_C4(cos, cos((0.9_4, 1.1_4)), &
    (1.0371677875518798828125_4,(-1.0462486743927001953125_4)))
  TEST_C4(cosh, cosh((1.1_4, 0.9_4)), &
    (1.0371677875518798828125_4,1.0462486743927001953125_4))
  TEST_C4(exp, exp((0.4_4, 0.8_4)), &
    (1.039364337921142578125_4,1.07016956806182861328125_4))
  TEST_C4(log, log((1.5_4, 2.5_4)), &
    (1.07003307342529296875_4,1.03037679195404052734375_4))
  TEST_C4(sin, sin((0.7_4, 1.1_4)), &
    (1.07488918304443359375_4,1.02155959606170654296875_4))
  TEST_C4(sinh, sinh((1.1_4, 0.7_4)), &
    (1.02155959606170654296875_4,1.07488918304443359375_4))
  TEST_C4(sqrt, sqrt((0.1_4, 2.1_4)), &
    (1.04937589168548583984375_4,1.00059473514556884765625_4))
  TEST_C4(tan, tan((1.1_4, 0.4_4)), &
    (1.07952976226806640625_4,1.1858270168304443359375_4))
  TEST_C4(tanh, tanh((0.4_4, 1.1_4)), &
    (1.1858270168304443359375_4,1.07952976226806640625_4))

! COMPLEX(8) tests.

  logical, parameter :: test_abs_c8 = abs(abs((1.1_8, 0.1_8)) &
    - 1.1045361017187260710414875575224868953227996826171875_8).LE.(eps4)
  TEST_C8(acos, acos((0.7_8, 1.1_8)), &
    (1.1125956244800556671492586247040890157222747802734375_8, &
      (-1.032837729564676454430127705563791096210479736328125_8)))
  TEST_C8(acosh, acosh((0.7_8, 1.1_8)), &
    (1.0328377295646762323855227805324830114841461181640625_8, &
      (1.1125956244800558891938635497353971004486083984375_8)))
  TEST_C8(asin, asin((1.4_8, 0.7_8)), &
    (1.010103922959187716656970223993994295597076416015625_8, &
      (1.088381716746653626870511288871057331562042236328125_8)))
  TEST_C8(asinh, asinh((0.7_8, 1.4_8)), &
    (1.088381716746653626870511288871057331562042236328125_8, &
      (1.0101039229591874946123652989626862108707427978515625_8)))
  TEST_C8(atan, atan((0.2_8, 1.1_8)), &
    (1.064697821069229721757665174663998186588287353515625_8, &
      (1.122159092433034910385458715609274804592132568359375_8)))
  TEST_C8(atanh, atanh((1.1_8, 0.2_8)), &
    (1.122159092433034910385458715609274804592132568359375_8, &
      (1.064697821069229721757665174663998186588287353515625_8)))
  TEST_C8(cmplx, cmplx(z'bff0000000000000', kind=8), (-1._8,0))
  TEST_C8(cos, cos((0.9_8, 1.1_8)), &
    (1.03716776530046761450876147137023508548736572265625_8, &
      (-1.0462486051241379758636185215436853468418121337890625_8)))
  TEST_C8(cosh, cosh((1.1_8, 0.9_8)), &
    (1.03716776530046761450876147137023508548736572265625_8, &
      (1.0462486051241379758636185215436853468418121337890625_8)))
  TEST_C8(exp, exp((0.4_8, 0.8_8)), &
    (1.039364276016479404773917849524877965450286865234375_8, &
      (1.0701695334073042520373064689920283854007720947265625_8)))
  TEST_C8(log, log((1.5_8, 2.5_8)), &
    (1.070033081748135384003717263112775981426239013671875_8, &
      (1.0303768265243125057395445764996111392974853515625_8)))
  TEST_C8(sin, sin((0.7_8, 1.1_8)), &
    (1.0748891638565509776270801012287847697734832763671875_8, &
      (1.0215595324907689178672853813623078167438507080078125_8)))
  TEST_C8(sinh, sinh((1.1_8, 0.7_8)), &
    (1.0215595324907689178672853813623078167438507080078125_8, &
      (1.0748891638565509776270801012287847697734832763671875_8)))
  TEST_C8(sqrt, sqrt((0.1_8, 2.1_8)), &
    (1.04937591075907210580453465809114277362823486328125_8, &
      (1.0005947241922830059472419228351358112816260614863494993187487125396728515625_8)))
  TEST_C8(tan, tan((1.1_8, 0.4_8)), &
    (1.07952982287592025301137255155481398105621337890625_8, &
      (1.1858270353667335061942367246956564486026763916015625_8)))
  TEST_C8(tanh, tanh((0.4_8, 1.1_8)), &
    (1.1858270353667335061942367246956564486026763916015625_8, &
      (1.07952982287592025301137255155481398105621337890625_8)))


  ! Only test a few REAL(2)/REAL(3) cases since they anyway use the real 4
  ! runtime mapping.
  TEST_R2(acos, acos(0.5_2), 1.046875_2)
  TEST_R2(atan2, atan2(1.5_2, 1._2), 9.8291015625e-1_2)

  TEST_R3(acos, acos(0.5_3), 1.046875_3)
  TEST_R3(atan2, atan2(1.3_2, 1._3), 9.140625e-1_3)

#ifdef TEST_LIBPGMATH
! Bessel functions and erfc_scaled can only be folded if libpgmath
! is used.
  TEST_R4(bessel_j0, bessel_j0(0.5_4), 0.938469827175140380859375_4)
  TEST_R4(bessel_j1, bessel_j1(1.8_4), 0.5815169811248779296875_4)
  TEST_R4(bessel_jn, bessel_jn(2, 3._4), 0.4860912859439849853515625_4)
  TEST_R4(bessel_y0, bessel_y0(2._4), 0.510375678539276123046875_4)
  TEST_R4(bessel_y1, bessel_y1(1._4), (-0.78121280670166015625_4))
  TEST_R4(bessel_yn, bessel_yn(2, 1.5_4), (-0.932193756103515625_4))
  TEST_R4(erfc_scaled, erfc_scaled(0.1_4), 0.8964569568634033203125_4)

  TEST_R8(bessel_j0, bessel_j0(0.5_8), &
    0.938469807240812858850631528184749186038970947265625_8)
  TEST_R8(bessel_j1, bessel_j1(1.8_8), &
    0.5815169517311653546443039886071346700191497802734375_8)
  TEST_R8(bessel_jn, bessel_jn(2, 3._8), &
    0.486091260585891082879328450871980749070644378662109375_8)
  TEST_R8(bessel_y0, bessel_y0(2._8), &
    0.51037567264974514902320379405864514410495758056640625_8)
  TEST_R8(bessel_y1, bessel_y1(1._8), &
    (-0.781212821300288684511770043172873556613922119140625_8))
  TEST_R8(bessel_yn, bessel_yn(2, 1.5_8), &
    (-0.93219375976297402797143831776338629424571990966796875_8))
   TEST_R8(erfc_scaled, erfc_scaled(0.1_8), &
    0.89645697996912654392787089818739332258701324462890625_8)

  real(4), parameter :: bessel_jn_transformational(*) = bessel_jn(1,3, 3.2_4)
  logical, parameter :: test_bessel_jn_shape = size(bessel_jn_transformational, 1).eq.3
  logical, parameter :: test_bessel_jn_t1 = bessel_jn_transformational(1).eq.bessel_jn(1, 3.2_4)
  logical, parameter :: test_bessel_jn_t2 = bessel_jn_transformational(2).eq.bessel_jn(2, 3.2_4)
  logical, parameter :: test_bessel_jn_t3 = bessel_jn_transformational(3).eq.bessel_jn(3, 3.2_4)
  real(4), parameter :: bessel_jn_empty(*) = bessel_jn(3,1, 3.2_4)
  logical, parameter :: test_bessel_jn_empty = size(bessel_jn_empty, 1).eq.0

  real(4), parameter :: bessel_yn_transformational(*) = bessel_yn(1,3, 1.6_4)
  logical, parameter :: test_bessel_yn_shape = size(bessel_yn_transformational, 1).eq.3
  logical, parameter :: test_bessel_yn_t1 = bessel_yn_transformational(1).eq.bessel_yn(1, 1.6_4)
  logical, parameter :: test_bessel_yn_t2 = bessel_yn_transformational(2).eq.bessel_yn(2, 1.6_4)
  logical, parameter :: test_bessel_yn_t3 = bessel_yn_transformational(3).eq.bessel_yn(3, 1.6_4)
  real(4), parameter :: bessel_yn_empty(*) = bessel_yn(3,1, 3.2_4)
  logical, parameter :: test_bessel_yn_empty = size(bessel_yn_empty, 1).eq.0
#endif

! Test exponentiation by real or complex folding (it is using host runtime)
  TEST_R4(pow, (0.5_4**3.14_4), 1.134398877620697021484375e-1_4)
  TEST_R8(pow, (0.5_8**3.14_8), &
    1.1343989441464509548840311481399112381041049957275390625e-1_8)
  TEST_C4(pow, ((0.5_4, 0.6_4)**(0.74_4, -1.1_4)), &
    (1.32234990596771240234375_4,1.73712027072906494140625_4))
  TEST_C8(pow, ((0.5_8, 0.6_8)**(0.74_8, -1.1_8)), &
    (1.3223499632715445262221010125358588993549346923828125_8, &
     1.7371201007364975854585509296157397329807281494140625_8))

! Extension specific intrinsic variants of ABS
  logical, parameter, test_babs1 = kind(babs(-1_1)) == 1
  logical, parameter, test_babs2 = babs(-1_1) == 1_1
  logical, parameter, test_iiabs1 = kind(iiabs(-1_2)) == 2
  logical, parameter, test_iiabs2 = iiabs(-1_2) == 1_2
  logical, parameter, test_jiabs1 = kind(jiabs(-1_4)) == 4
  logical, parameter, test_jiabs2 = jiabs(-1_4) == 1_4
  logical, parameter, test_kiabs1 = kind(kiabs(-1_8)) == 8
  logical, parameter, test_kiabs2 = kiabs(-1_8) == 1_8
  logical, parameter, test_zabs1 = kind(zabs((3._8,4._8))) == 8
  logical, parameter, test_zabs2 = zabs((3._8,4._8)) == 5_8
  logical, parameter, test_cdabs1 = kind(cdabs((3._8,4._8))) == kind(1.d0)
  logical, parameter, test_cdabs2 = cdabs((3._8,4._8)) == real(5, kind(1.d0))

end
