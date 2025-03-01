! RUN: bbc -emit-hlfir -o - %s | FileCheck %s

! CHECK-LABEL: c.func @_QQmain
program p
  use ieee_arithmetic, only: ieee_int, ieee_rint
  use ieee_arithmetic, only: ieee_value, ieee_positive_inf
  use ieee_arithmetic, only: ieee_to_zero, ieee_nearest, ieee_up, ieee_away

  ! CHECK:     %[[V_10:[0-9]+]] = fir.alloca i32 {bindc_name = "n", uniq_name = "_QFEn"}
  ! CHECK:     %[[V_11:[0-9]+]]:2 = hlfir.declare %[[V_10]] {uniq_name = "_QFEn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  ! CHECK:     %[[V_12:[0-9]+]] = fir.alloca i128 {bindc_name = "n16", uniq_name = "_QFEn16"}
  ! CHECK:     %[[V_13:[0-9]+]]:2 = hlfir.declare %[[V_12]] {uniq_name = "_QFEn16"} : (!fir.ref<i128>) -> (!fir.ref<i128>, !fir.ref<i128>)
  ! CHECK:     %[[V_14:[0-9]+]] = fir.alloca i16 {bindc_name = "n2", uniq_name = "_QFEn2"}
  ! CHECK:     %[[V_15:[0-9]+]]:2 = hlfir.declare %[[V_14]] {uniq_name = "_QFEn2"} : (!fir.ref<i16>) -> (!fir.ref<i16>, !fir.ref<i16>)
  ! CHECK:     %[[V_16:[0-9]+]] = fir.alloca i64 {bindc_name = "n8", uniq_name = "_QFEn8"}
  ! CHECK:     %[[V_17:[0-9]+]]:2 = hlfir.declare %[[V_16]] {uniq_name = "_QFEn8"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
  ! CHECK:     %[[V_18:[0-9]+]] = fir.alloca f32 {bindc_name = "x", uniq_name = "_QFEx"}
  ! CHECK:     %[[V_19:[0-9]+]]:2 = hlfir.declare %[[V_18]] {uniq_name = "_QFEx"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  ! CHECK:     %[[V_20:[0-9]+]] = fir.alloca f16 {bindc_name = "x2", uniq_name = "_QFEx2"}
  ! CHECK:     %[[V_21:[0-9]+]]:2 = hlfir.declare %[[V_20]] {uniq_name = "_QFEx2"} : (!fir.ref<f16>) -> (!fir.ref<f16>, !fir.ref<f16>)
  ! CHECK:     %[[V_22:[0-9]+]] = fir.alloca bf16 {bindc_name = "x3", uniq_name = "_QFEx3"}
  ! CHECK:     %[[V_23:[0-9]+]]:2 = hlfir.declare %[[V_22]] {uniq_name = "_QFEx3"} : (!fir.ref<bf16>) -> (!fir.ref<bf16>, !fir.ref<bf16>)
  ! CHECK:     %[[V_24:[0-9]+]] = fir.alloca f32 {bindc_name = "x8", uniq_name = "_QFEx8"}
  ! CHECK:     %[[V_25:[0-9]+]]:2 = hlfir.declare %[[V_24]] {uniq_name = "_QFEx8"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  ! CHECK:     %[[V_26:[0-9]+]] = fir.alloca f32 {bindc_name = "y", uniq_name = "_QFEy"}
  ! CHECK:     %[[V_27:[0-9]+]]:2 = hlfir.declare %[[V_26]] {uniq_name = "_QFEy"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  ! CHECK:     %[[V_28:[0-9]+]] = fir.alloca f16 {bindc_name = "y2", uniq_name = "_QFEy2"}
  ! CHECK:     %[[V_29:[0-9]+]]:2 = hlfir.declare %[[V_28]] {uniq_name = "_QFEy2"} : (!fir.ref<f16>) -> (!fir.ref<f16>, !fir.ref<f16>)
  ! CHECK:     %[[V_30:[0-9]+]] = fir.alloca bf16 {bindc_name = "y3", uniq_name = "_QFEy3"}
  ! CHECK:     %[[V_31:[0-9]+]]:2 = hlfir.declare %[[V_30]] {uniq_name = "_QFEy3"} : (!fir.ref<bf16>) -> (!fir.ref<bf16>, !fir.ref<bf16>)
  ! CHECK:     %[[V_32:[0-9]+]] = fir.alloca f32 {bindc_name = "y8", uniq_name = "_QFEy8"}
  ! CHECK:     %[[V_33:[0-9]+]]:2 = hlfir.declare %[[V_32]] {uniq_name = "_QFEy8"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  integer(2) n2
  integer(8) n8
  integer(16) n16
  real(2) x2, y2
  real(3) x3, y3

  ! CHECK:     hlfir.assign %cst{{[_0-9]*}} to %[[V_19]]#0 : f32, !fir.ref<f32>
  x = -200.7

  ! CHECK:     %[[V_34:[0-9]+]] = fir.address_of(@_QQro._QM__fortran_builtinsT__builtin_ieee_round_type.0) : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_round_type{_QM__fortran_builtinsT__builtin_ieee_round_type.mode:i8}>>
  ! CHECK:     %[[V_35:[0-9]+]]:2 = hlfir.declare %[[V_34]]
  ! CHECK:     %[[V_36:[0-9]+]] = fir.load %[[V_19]]#0 : !fir.ref<f32>
  ! CHECK:     %[[V_37:[0-9]+]] = fir.call @llvm.get.rounding() fastmath<contract> : () -> i32
  ! CHECK:     %[[V_39:[0-9]+]] = fir.coordinate_of %[[V_35]]#1, _QM__fortran_builtinsT__builtin_ieee_round_type.mode : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_round_type{_QM__fortran_builtinsT__builtin_ieee_round_type.mode:i8}>>) -> !fir.ref<i8>
  ! CHECK:     %[[V_40:[0-9]+]] = fir.load %[[V_39]] : !fir.ref<i8>
  ! CHECK:     %[[V_41:[0-9]+]] = arith.shli %c-1{{.*}}, %c2{{.*}} : i8
  ! CHECK:     %[[V_42:[0-9]+]] = arith.andi %[[V_40]], %[[V_41]] : i8
  ! CHECK:     %[[V_43:[0-9]+]] = arith.cmpi eq, %[[V_42]], %c0{{.*}} : i8
  ! CHECK:     %[[V_44:[0-9]+]] = arith.select %[[V_43]], %[[V_40]], %c1{{.*}} : i8
  ! CHECK:     %[[V_45:[0-9]+]] = fir.convert %[[V_44]] : (i8) -> i32
  ! CHECK:     fir.call @llvm.set.rounding(%[[V_45]]) fastmath<contract> : (i32) -> ()
  ! CHECK:     %[[V_46:[0-9]+]] = fir.call @llvm.nearbyint.f32(%[[V_36]]) fastmath<contract> : (f32) -> f32
  ! CHECK:     %[[V_47:[0-9]+]] = fir.convert %[[V_46]] : (f32) -> f32
  ! CHECK:     fir.call @llvm.set.rounding(%[[V_37]]) fastmath<contract> : (i32) -> ()
  ! CHECK:     hlfir.assign %[[V_47]] to %[[V_27]]#0 : f32, !fir.ref<f32>
  y = ieee_rint(x, ieee_nearest)

  ! CHECK:     %[[V_48:[0-9]+]] = fir.address_of(@_QQro._QM__fortran_builtinsT__builtin_ieee_round_type.0) : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_round_type{_QM__fortran_builtinsT__builtin_ieee_round_type.mode:i8}>>
  ! CHECK:     %[[V_49:[0-9]+]]:2 = hlfir.declare %[[V_48]]
  ! CHECK:     %[[V_50:[0-9]+]] = fir.load %[[V_19]]#0 : !fir.ref<f32>
  ! CHECK:     %[[V_51:[0-9]+]] = fir.call @llvm.get.rounding() fastmath<contract> : () -> i32
  ! CHECK:     %[[V_53:[0-9]+]] = fir.coordinate_of %[[V_49]]#1, _QM__fortran_builtinsT__builtin_ieee_round_type.mode : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_round_type{_QM__fortran_builtinsT__builtin_ieee_round_type.mode:i8}>>) -> !fir.ref<i8>
  ! CHECK:     %[[V_54:[0-9]+]] = fir.load %[[V_53]] : !fir.ref<i8>
  ! CHECK:     %[[V_55:[0-9]+]] = arith.shli %c-1{{.*}}, %c2{{.*}} : i8
  ! CHECK:     %[[V_56:[0-9]+]] = arith.andi %[[V_54]], %[[V_55]] : i8
  ! CHECK:     %[[V_57:[0-9]+]] = arith.cmpi eq, %[[V_56]], %c0{{.*}} : i8
  ! CHECK:     %[[V_58:[0-9]+]] = arith.select %[[V_57]], %[[V_54]], %c1{{.*}} : i8
  ! CHECK:     %[[V_59:[0-9]+]] = fir.convert %[[V_58]] : (i8) -> i32
  ! CHECK:     fir.call @llvm.set.rounding(%[[V_59]]) fastmath<contract> : (i32) -> ()
  ! CHECK:     %[[V_60:[0-9]+]] = fir.call @llvm.nearbyint.f32(%[[V_50]]) fastmath<contract> : (f32) -> f32
  ! CHECK:     %[[V_61:[0-9]+]] = fir.convert %[[V_60]] : (f32) -> f32
  ! CHECK:     fir.call @llvm.set.rounding(%[[V_51]]) fastmath<contract> : (i32) -> ()
  ! CHECK:     %[[V_62:[0-9]+]] = fir.convert %c-2147483648{{.*}} : (i32) -> f32
  ! CHECK:     %[[V_63:[0-9]+]] = arith.negf %[[V_62]] fastmath<contract> : f32
  ! CHECK:     %[[V_64:[0-9]+]] = arith.cmpf oge, %[[V_61]], %[[V_62]] fastmath<contract> : f32
  ! CHECK:     %[[V_65:[0-9]+]] = arith.cmpf olt, %[[V_61]], %[[V_63]] fastmath<contract> : f32
  ! CHECK:     %[[V_66:[0-9]+]] = arith.andi %[[V_64]], %[[V_65]] : i1
  ! CHECK:     %[[V_67:[0-9]+]] = fir.if %[[V_66]] -> (i32) {
  ! CHECK:       %[[V_163:[0-9]+]] = arith.cmpf one, %[[V_50]], %[[V_61]] fastmath<contract> : f32
  ! CHECK:       fir.if %[[V_163]] {
  ! CHECK:         %[[V_165:[0-9]+]] = fir.call @_FortranAMapException(%c32{{.*}}) fastmath<contract> : (i32) -> i32
  ! CHECK:         %[[V_166:[0-9]+]] = fir.call @feraiseexcept(%[[V_165]]) fastmath<contract> : (i32) -> i32
  ! CHECK:       }
  ! CHECK:       %[[V_164:[0-9]+]] = fir.convert %[[V_61]] : (f32) -> i32
  ! CHECK:       fir.result %[[V_164]] : i32
  ! CHECK:     } else {
  ! CHECK:       %[[V_163:[0-9]+]] = fir.call @_FortranAMapException(%c1{{.*}}) fastmath<contract> : (i32) -> i32
  ! CHECK:       %[[V_164:[0-9]+]] = fir.call @feraiseexcept(%[[V_163]]) fastmath<contract> : (i32) -> i32
  ! CHECK:       %[[V_165:[0-9]+]] = arith.select %[[V_64]], %c2147483647{{.*}}, %c-2147483648{{.*}} : i32
  ! CHECK:       fir.result %[[V_165]] : i32
  ! CHECK:     }
  ! CHECK:     hlfir.assign %[[V_67]] to %[[V_11]]#0 : i32, !fir.ref<i32>
  n = ieee_int(x, ieee_nearest)
! print*, x, ' -> ', y, ' -> ', n

  ! CHECK:     hlfir.assign %cst{{[_0-9]*}} to %[[V_21]]#0 : f16, !fir.ref<f16>
  x2 = huge(x2)

  ! CHECK:     %[[V_68:[0-9]+]] = fir.address_of(@_QQro._QM__fortran_builtinsT__builtin_ieee_round_type.1) : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_round_type{_QM__fortran_builtinsT__builtin_ieee_round_type.mode:i8}>>
  ! CHECK:     %[[V_69:[0-9]+]]:2 = hlfir.declare %[[V_68]]
  ! CHECK:     %[[V_70:[0-9]+]] = fir.load %[[V_21]]#0 : !fir.ref<f16>
  ! CHECK:     %[[V_71:[0-9]+]] = fir.call @llvm.get.rounding() fastmath<contract> : () -> i32
  ! CHECK:     %[[V_73:[0-9]+]] = fir.coordinate_of %[[V_69]]#1, _QM__fortran_builtinsT__builtin_ieee_round_type.mode : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_round_type{_QM__fortran_builtinsT__builtin_ieee_round_type.mode:i8}>>) -> !fir.ref<i8>
  ! CHECK:     %[[V_74:[0-9]+]] = fir.load %[[V_73]] : !fir.ref<i8>
  ! CHECK:     %[[V_75:[0-9]+]] = arith.shli %c-1{{.*}}, %c2{{.*}} : i8
  ! CHECK:     %[[V_76:[0-9]+]] = arith.andi %[[V_74]], %[[V_75]] : i8
  ! CHECK:     %[[V_77:[0-9]+]] = arith.cmpi eq, %[[V_76]], %c0{{.*}} : i8
  ! CHECK:     %[[V_78:[0-9]+]] = arith.select %[[V_77]], %[[V_74]], %c1{{.*}} : i8
  ! CHECK:     %[[V_79:[0-9]+]] = fir.convert %[[V_78]] : (i8) -> i32
  ! CHECK:     fir.call @llvm.set.rounding(%[[V_79]]) fastmath<contract> : (i32) -> ()
  ! CHECK:     %[[V_80:[0-9]+]] = fir.convert %[[V_70]] : (f16) -> f32
  ! CHECK:     %[[V_81:[0-9]+]] = fir.call @llvm.nearbyint.f32(%[[V_80]]) fastmath<contract> : (f32) -> f32
  ! CHECK:     %[[V_82:[0-9]+]] = fir.convert %[[V_81]] : (f32) -> f16
  ! CHECK:     fir.call @llvm.set.rounding(%[[V_71]]) fastmath<contract> : (i32) -> ()
  ! CHECK:     hlfir.assign %[[V_82]] to %[[V_29]]#0 : f16, !fir.ref<f16>
  y2 = ieee_rint(x2, ieee_up)

  ! CHECK:     %[[V_83:[0-9]+]] = fir.address_of(@_QQro._QM__fortran_builtinsT__builtin_ieee_round_type.1) : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_round_type{_QM__fortran_builtinsT__builtin_ieee_round_type.mode:i8}>>
  ! CHECK:     %[[V_84:[0-9]+]]:2 = hlfir.declare %[[V_83]]
  ! CHECK:     %[[V_85:[0-9]+]] = fir.load %[[V_21]]#0 : !fir.ref<f16>
  ! CHECK:     %[[V_86:[0-9]+]] = fir.call @llvm.get.rounding() fastmath<contract> : () -> i32
  ! CHECK:     %[[V_88:[0-9]+]] = fir.coordinate_of %[[V_84]]#1, _QM__fortran_builtinsT__builtin_ieee_round_type.mode : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_round_type{_QM__fortran_builtinsT__builtin_ieee_round_type.mode:i8}>>) -> !fir.ref<i8>
  ! CHECK:     %[[V_89:[0-9]+]] = fir.load %[[V_88]] : !fir.ref<i8>
  ! CHECK:     %[[V_90:[0-9]+]] = arith.shli %c-1{{.*}}, %c2{{.*}} : i8
  ! CHECK:     %[[V_91:[0-9]+]] = arith.andi %[[V_89]], %[[V_90]] : i8
  ! CHECK:     %[[V_92:[0-9]+]] = arith.cmpi eq, %[[V_91]], %c0{{.*}} : i8
  ! CHECK:     %[[V_93:[0-9]+]] = arith.select %[[V_92]], %[[V_89]], %c1{{.*}} : i8
  ! CHECK:     %[[V_94:[0-9]+]] = fir.convert %[[V_93]] : (i8) -> i32
  ! CHECK:     fir.call @llvm.set.rounding(%[[V_94]]) fastmath<contract> : (i32) -> ()
  ! CHECK:     %[[V_95:[0-9]+]] = fir.convert %[[V_85]] : (f16) -> f32
  ! CHECK:     %[[V_96:[0-9]+]] = fir.call @llvm.nearbyint.f32(%[[V_95]]) fastmath<contract> : (f32) -> f32
  ! CHECK:     %[[V_97:[0-9]+]] = fir.convert %[[V_96]] : (f32) -> f16
  ! CHECK:     fir.call @llvm.set.rounding(%[[V_86]]) fastmath<contract> : (i32) -> ()
  ! CHECK:     %[[V_98:[0-9]+]] = fir.convert %c-9223372036854775808{{.*}} : (i64) -> f16
  ! CHECK:     %[[V_99:[0-9]+]] = arith.negf %[[V_98]] fastmath<contract> : f16
  ! CHECK:     %[[V_100:[0-9]+]] = arith.cmpf oge, %[[V_97]], %[[V_98]] fastmath<contract> : f16
  ! CHECK:     %[[V_101:[0-9]+]] = arith.cmpf olt, %[[V_97]], %[[V_99]] fastmath<contract> : f16
  ! CHECK:     %[[V_102:[0-9]+]] = arith.andi %[[V_100]], %[[V_101]] : i1
  ! CHECK:     %[[V_103:[0-9]+]] = fir.if %[[V_102]] -> (i64) {
  ! CHECK:       %[[V_163:[0-9]+]] = arith.cmpf one, %[[V_85]], %[[V_97]] fastmath<contract> : f16
  ! CHECK:       fir.if %[[V_163]] {
  ! CHECK:         %[[V_165:[0-9]+]] = fir.call @_FortranAMapException(%c32{{.*}}) fastmath<contract> : (i32) -> i32
  ! CHECK:         %[[V_166:[0-9]+]] = fir.call @feraiseexcept(%[[V_165]]) fastmath<contract> : (i32) -> i32
  ! CHECK:       }
  ! CHECK:       %[[V_164:[0-9]+]] = fir.convert %[[V_97]] : (f16) -> i64
  ! CHECK:       fir.result %[[V_164]] : i64
  ! CHECK:     } else {
  ! CHECK:       %[[V_163:[0-9]+]] = fir.call @_FortranAMapException(%c1{{.*}}) fastmath<contract> : (i32) -> i32
  ! CHECK:       %[[V_164:[0-9]+]] = fir.call @feraiseexcept(%[[V_163]]) fastmath<contract> : (i32) -> i32
  ! CHECK:       %[[V_165:[0-9]+]] = arith.select %[[V_100]], %c9223372036854775807{{.*}}, %c-9223372036854775808{{.*}} : i64
  ! CHECK:       fir.result %[[V_165]] : i64
  ! CHECK:     }
  ! CHECK:     hlfir.assign %[[V_103]] to %[[V_17]]#0 : i64, !fir.ref<i64>
  n8 = ieee_int(x2, ieee_up, 8)

! print*, x2, ' -> ', y2, ' -> ', n8

  ! CHECK:     hlfir.assign %cst{{[_0-9]*}} to %[[V_23]]#0 : bf16, !fir.ref<bf16>
  x3 = -0.

  ! CHECK:     %[[V_104:[0-9]+]] = fir.load %[[V_23]]#0 : !fir.ref<bf16>
  ! CHECK:     %[[V_105:[0-9]+]] = fir.convert %[[V_104]] : (bf16) -> f32
  ! CHECK:     %[[V_106:[0-9]+]] = fir.call @llvm.nearbyint.f32(%[[V_105]]) fastmath<contract> : (f32) -> f32
  ! CHECK:     %[[V_107:[0-9]+]] = fir.convert %[[V_106]] : (f32) -> bf16
  ! CHECK:     %[[V_108:[0-9]+]] = arith.cmpf one, %[[V_104]], %[[V_107]] fastmath<contract> : bf16
  ! CHECK:     fir.if %[[V_108]] {
  ! CHECK:       %[[V_163:[0-9]+]] = fir.call @_FortranAMapException(%c32{{.*}}) fastmath<contract> : (i32) -> i32
  ! CHECK:       %[[V_164:[0-9]+]] = fir.call @feraiseexcept(%[[V_163]]) fastmath<contract> : (i32) -> i32
  ! CHECK:     }
  ! CHECK:     hlfir.assign %[[V_107]] to %[[V_31]]#0 : bf16, !fir.ref<bf16>
  y3 = ieee_rint(x3)

  ! CHECK:     %[[V_109:[0-9]+]] = fir.address_of(@_QQro._QM__fortran_builtinsT__builtin_ieee_round_type.2) : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_round_type{_QM__fortran_builtinsT__builtin_ieee_round_type.mode:i8}>>
  ! CHECK:     %[[V_110:[0-9]+]]:2 = hlfir.declare %[[V_109]]
  ! CHECK:     %[[V_111:[0-9]+]] = fir.load %[[V_23]]#0 : !fir.ref<bf16>
  ! CHECK:     %[[V_112:[0-9]+]] = fir.call @llvm.get.rounding() fastmath<contract> : () -> i32
  ! CHECK:     %[[V_114:[0-9]+]] = fir.coordinate_of %[[V_110]]#1, _QM__fortran_builtinsT__builtin_ieee_round_type.mode : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_round_type{_QM__fortran_builtinsT__builtin_ieee_round_type.mode:i8}>>) -> !fir.ref<i8>
  ! CHECK:     %[[V_115:[0-9]+]] = fir.load %[[V_114]] : !fir.ref<i8>
  ! CHECK:     %[[V_116:[0-9]+]] = arith.shli %c-1{{.*}}, %c2{{.*}} : i8
  ! CHECK:     %[[V_117:[0-9]+]] = arith.andi %[[V_115]], %[[V_116]] : i8
  ! CHECK:     %[[V_118:[0-9]+]] = arith.cmpi eq, %[[V_117]], %c0{{.*}} : i8
  ! CHECK:     %[[V_119:[0-9]+]] = arith.select %[[V_118]], %[[V_115]], %c1{{.*}} : i8
  ! CHECK:     %[[V_120:[0-9]+]] = fir.convert %[[V_119]] : (i8) -> i32
  ! CHECK:     fir.call @llvm.set.rounding(%[[V_120]]) fastmath<contract> : (i32) -> ()
  ! CHECK:     %[[V_121:[0-9]+]] = fir.convert %[[V_111]] : (bf16) -> f32
  ! CHECK:     %[[V_122:[0-9]+]] = fir.call @llvm.nearbyint.f32(%[[V_121]]) fastmath<contract> : (f32) -> f32
  ! CHECK:     %[[V_123:[0-9]+]] = fir.convert %[[V_122]] : (f32) -> bf16
  ! CHECK:     fir.call @llvm.set.rounding(%[[V_112]]) fastmath<contract> : (i32) -> ()
  ! CHECK:     %[[V_124:[0-9]+]] = fir.convert %c-170141183460469231731687303715884105728{{.*}} : (i128) -> bf16
  ! CHECK:     %[[V_125:[0-9]+]] = arith.negf %[[V_124]] fastmath<contract> : bf16
  ! CHECK:     %[[V_126:[0-9]+]] = arith.cmpf oge, %[[V_123]], %[[V_124]] fastmath<contract> : bf16
  ! CHECK:     %[[V_127:[0-9]+]] = arith.cmpf olt, %[[V_123]], %[[V_125]] fastmath<contract> : bf16
  ! CHECK:     %[[V_128:[0-9]+]] = arith.andi %[[V_126]], %[[V_127]] : i1
  ! CHECK:     %[[V_129:[0-9]+]] = fir.if %[[V_128]] -> (i128) {
  ! CHECK:       %[[V_163:[0-9]+]] = arith.cmpf one, %[[V_111]], %[[V_123]] fastmath<contract> : bf16
  ! CHECK:       fir.if %[[V_163]] {
  ! CHECK:         %[[V_165:[0-9]+]] = fir.call @_FortranAMapException(%c32{{.*}}) fastmath<contract> : (i32) -> i32
  ! CHECK:         %[[V_166:[0-9]+]] = fir.call @feraiseexcept(%[[V_165]]) fastmath<contract> : (i32) -> i32
  ! CHECK:       }
  ! CHECK:       %[[V_164:[0-9]+]] = fir.convert %[[V_123]] : (bf16) -> i128
  ! CHECK:       fir.result %[[V_164]] : i128
  ! CHECK:     } else {
  ! CHECK:       %[[V_163:[0-9]+]] = fir.call @_FortranAMapException(%c1{{.*}}) fastmath<contract> : (i32) -> i32
  ! CHECK:       %[[V_164:[0-9]+]] = fir.call @feraiseexcept(%[[V_163]]) fastmath<contract> : (i32) -> i32
  ! CHECK:       %[[V_165:[0-9]+]] = arith.select %[[V_126]], %c170141183460469231731687303715884105727{{.*}}, %c-170141183460469231731687303715884105728{{.*}} : i128
  ! CHECK:       fir.result %[[V_165]] : i128
  ! CHECK:     }
  ! CHECK:     hlfir.assign %[[V_129]] to %[[V_13]]#0 : i128, !fir.ref<i128>
  n16 = ieee_int(x3, ieee_away, 16) ! ieee_away is not supported, treated as ieee_nearest

! print*, x3, ' -> ', y3, ' -> ', n16

  ! CHECK:     %[[V_130:[0-9]+]] = fir.address_of(@_QQro._QMieee_arithmeticTieee_class_type.3) : !fir.ref<!fir.type<_QMieee_arithmeticTieee_class_type{_QMieee_arithmeticTieee_class_type.which:i8}>>
  ! CHECK:     %[[V_131:[0-9]+]]:2 = hlfir.declare %[[V_130]]
  ! CHECK:     %[[V_133:[0-9]+]] = fir.coordinate_of %[[V_131]]#1, _QMieee_arithmeticTieee_class_type.which : (!fir.ref<!fir.type<_QMieee_arithmeticTieee_class_type{_QMieee_arithmeticTieee_class_type.which:i8}>>) -> !fir.ref<i8>
  ! CHECK:     %[[V_134:[0-9]+]] = fir.load %[[V_133]] : !fir.ref<i8>
  ! CHECK:     %[[V_135:[0-9]+]] = fir.address_of(@_FortranAIeeeValueTable_4) : !fir.ref<!fir.array<12xi32>>
  ! CHECK:     %[[V_136:[0-9]+]] = fir.coordinate_of %[[V_135]], %[[V_134]] : (!fir.ref<!fir.array<12xi32>>, i8) -> !fir.ref<i32>
  ! CHECK:     %[[V_137:[0-9]+]] = fir.load %[[V_136]] : !fir.ref<i32>
  ! CHECK:     %[[V_138:[0-9]+]] = arith.bitcast %[[V_137]] : i32 to f32
  ! CHECK:     hlfir.assign %[[V_138]] to %[[V_25]]#0 : f32, !fir.ref<f32>
  x8 = ieee_value(x8, ieee_positive_inf)

  ! CHECK:     %[[V_139:[0-9]+]] = fir.load %[[V_25]]#0 : !fir.ref<f32>
  ! CHECK:     %[[V_140:[0-9]+]] = fir.call @llvm.nearbyint.f32(%[[V_139]]) fastmath<contract> : (f32) -> f32
  ! CHECK:     %[[V_141:[0-9]+]] = fir.convert %[[V_140]] : (f32) -> f32
  ! CHECK:     %[[V_142:[0-9]+]] = arith.cmpf one, %[[V_139]], %[[V_141]] fastmath<contract> : f32
  ! CHECK:     fir.if %[[V_142]] {
  ! CHECK:       %[[V_163:[0-9]+]] = fir.call @_FortranAMapException(%c32{{.*}}) fastmath<contract> : (i32) -> i32
  ! CHECK:       %[[V_164:[0-9]+]] = fir.call @feraiseexcept(%[[V_163]]) fastmath<contract> : (i32) -> i32
  ! CHECK:     }
  ! CHECK:     hlfir.assign %[[V_141]] to %[[V_33]]#0 : f32, !fir.ref<f32>
  y8 = ieee_rint(x8)

  ! CHECK:     %[[V_143:[0-9]+]] = fir.address_of(@_QQro._QM__fortran_builtinsT__builtin_ieee_round_type.4) : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_round_type{_QM__fortran_builtinsT__builtin_ieee_round_type.mode:i8}>>
  ! CHECK:     %[[V_144:[0-9]+]]:2 = hlfir.declare %[[V_143]]
  ! CHECK:     %[[V_145:[0-9]+]] = fir.load %[[V_25]]#0 : !fir.ref<f32>
  ! CHECK:     %[[V_146:[0-9]+]] = fir.call @llvm.get.rounding() fastmath<contract> : () -> i32
  ! CHECK:     %[[V_148:[0-9]+]] = fir.coordinate_of %[[V_144]]#1, _QM__fortran_builtinsT__builtin_ieee_round_type.mode : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_round_type{_QM__fortran_builtinsT__builtin_ieee_round_type.mode:i8}>>) -> !fir.ref<i8>
  ! CHECK:     %[[V_149:[0-9]+]] = fir.load %[[V_148]] : !fir.ref<i8>
  ! CHECK:     %[[V_150:[0-9]+]] = arith.shli %c-1{{.*}}, %c2{{.*}} : i8
  ! CHECK:     %[[V_151:[0-9]+]] = arith.andi %[[V_149]], %[[V_150]] : i8
  ! CHECK:     %[[V_152:[0-9]+]] = arith.cmpi eq, %[[V_151]], %c0{{.*}} : i8
  ! CHECK:     %[[V_153:[0-9]+]] = arith.select %[[V_152]], %[[V_149]], %c1{{.*}} : i8
  ! CHECK:     %[[V_154:[0-9]+]] = fir.convert %[[V_153]] : (i8) -> i32
  ! CHECK:     fir.call @llvm.set.rounding(%[[V_154]]) fastmath<contract> : (i32) -> ()
  ! CHECK:     %[[V_155:[0-9]+]] = fir.call @llvm.nearbyint.f32(%[[V_145]]) fastmath<contract> : (f32) -> f32
  ! CHECK:     %[[V_156:[0-9]+]] = fir.convert %[[V_155]] : (f32) -> f32
  ! CHECK:     fir.call @llvm.set.rounding(%[[V_146]]) fastmath<contract> : (i32) -> ()
  ! CHECK:     %[[V_157:[0-9]+]] = fir.convert %c-32768{{.*}} : (i16) -> f32
  ! CHECK:     %[[V_158:[0-9]+]] = arith.negf %[[V_157]] fastmath<contract> : f32
  ! CHECK:     %[[V_159:[0-9]+]] = arith.cmpf oge, %[[V_156]], %[[V_157]] fastmath<contract> : f32
  ! CHECK:     %[[V_160:[0-9]+]] = arith.cmpf olt, %[[V_156]], %[[V_158]] fastmath<contract> : f32
  ! CHECK:     %[[V_161:[0-9]+]] = arith.andi %[[V_159]], %[[V_160]] : i1
  ! CHECK:     %[[V_162:[0-9]+]] = fir.if %[[V_161]] -> (i16) {
  ! CHECK:       %[[V_163:[0-9]+]] = arith.cmpf one, %[[V_145]], %[[V_156]] fastmath<contract> : f32
  ! CHECK:       fir.if %[[V_163]] {
  ! CHECK:         %[[V_165:[0-9]+]] = fir.call @_FortranAMapException(%c32{{.*}}) fastmath<contract> : (i32) -> i32
  ! CHECK:         %[[V_166:[0-9]+]] = fir.call @feraiseexcept(%[[V_165]]) fastmath<contract> : (i32) -> i32
  ! CHECK:       }
  ! CHECK:       %[[V_164:[0-9]+]] = fir.convert %[[V_156]] : (f32) -> i16
  ! CHECK:       fir.result %[[V_164]] : i16
  ! CHECK:     } else {
  ! CHECK:       %[[V_163:[0-9]+]] = fir.call @_FortranAMapException(%c1{{.*}}) fastmath<contract> : (i32) -> i32
  ! CHECK:       %[[V_164:[0-9]+]] = fir.call @feraiseexcept(%[[V_163]]) fastmath<contract> : (i32) -> i32
  ! CHECK:       %[[V_165:[0-9]+]] = arith.select %[[V_159]], %c32767{{.*}}, %c-32768{{.*}} : i16
  ! CHECK:       fir.result %[[V_165]] : i16
  ! CHECK:     }
  ! CHECK:     hlfir.assign %[[V_162]] to %[[V_15]]#0 : i16, !fir.ref<i16>
  n2 = ieee_int(x8, ieee_to_zero, 2)

! print*, x8, ' -> ', y8, ' -> ', n2
end
