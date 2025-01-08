! RUN: bbc -emit-fir -o - %s | FileCheck %s

! CHECK-LABEL: c.func @_QQmain
program p
  use ieee_arithmetic, only: ieee_int, ieee_rint
  use ieee_arithmetic, only: ieee_value, ieee_positive_inf
  use ieee_arithmetic, only: ieee_to_zero, ieee_nearest, ieee_up, ieee_away

  ! CHECK:     %[[V_10:[0-9]+]] = fir.alloca i32 {bindc_name = "n", uniq_name = "_QFEn"}
  ! CHECK:     %[[V_11:[0-9]+]] = fir.declare %[[V_10]] {uniq_name = "_QFEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
  ! CHECK:     %[[V_12:[0-9]+]] = fir.alloca i128 {bindc_name = "n16", uniq_name = "_QFEn16"}
  ! CHECK:     %[[V_13:[0-9]+]] = fir.declare %[[V_12]] {uniq_name = "_QFEn16"} : (!fir.ref<i128>) -> !fir.ref<i128>
  ! CHECK:     %[[V_14:[0-9]+]] = fir.alloca i16 {bindc_name = "n2", uniq_name = "_QFEn2"}
  ! CHECK:     %[[V_15:[0-9]+]] = fir.declare %[[V_14]] {uniq_name = "_QFEn2"} : (!fir.ref<i16>) -> !fir.ref<i16>
  ! CHECK:     %[[V_16:[0-9]+]] = fir.alloca i64 {bindc_name = "n8", uniq_name = "_QFEn8"}
  ! CHECK:     %[[V_17:[0-9]+]] = fir.declare %[[V_16]] {uniq_name = "_QFEn8"} : (!fir.ref<i64>) -> !fir.ref<i64>
  ! CHECK:     %[[V_18:[0-9]+]] = fir.alloca f32 {bindc_name = "x", uniq_name = "_QFEx"}
  ! CHECK:     %[[V_19:[0-9]+]] = fir.declare %[[V_18]] {uniq_name = "_QFEx"} : (!fir.ref<f32>) -> !fir.ref<f32>
  ! CHECK:     %[[V_20:[0-9]+]] = fir.alloca f16 {bindc_name = "x2", uniq_name = "_QFEx2"}
  ! CHECK:     %[[V_21:[0-9]+]] = fir.declare %[[V_20]] {uniq_name = "_QFEx2"} : (!fir.ref<f16>) -> !fir.ref<f16>
  ! CHECK:     %[[V_22:[0-9]+]] = fir.alloca bf16 {bindc_name = "x3", uniq_name = "_QFEx3"}
  ! CHECK:     %[[V_23:[0-9]+]] = fir.declare %[[V_22]] {uniq_name = "_QFEx3"} : (!fir.ref<bf16>) -> !fir.ref<bf16>
  ! CHECK:     %[[V_24:[0-9]+]] = fir.alloca f32 {bindc_name = "x8", uniq_name = "_QFEx8"}
  ! CHECK:     %[[V_25:[0-9]+]] = fir.declare %[[V_24]] {uniq_name = "_QFEx8"} : (!fir.ref<f32>) -> !fir.ref<f32>
  ! CHECK:     %[[V_26:[0-9]+]] = fir.alloca f32 {bindc_name = "y", uniq_name = "_QFEy"}
  ! CHECK:     %[[V_27:[0-9]+]] = fir.declare %[[V_26]] {uniq_name = "_QFEy"} : (!fir.ref<f32>) -> !fir.ref<f32>
  ! CHECK:     %[[V_28:[0-9]+]] = fir.alloca f16 {bindc_name = "y2", uniq_name = "_QFEy2"}
  ! CHECK:     %[[V_29:[0-9]+]] = fir.declare %[[V_28]] {uniq_name = "_QFEy2"} : (!fir.ref<f16>) -> !fir.ref<f16>
  ! CHECK:     %[[V_30:[0-9]+]] = fir.alloca bf16 {bindc_name = "y3", uniq_name = "_QFEy3"}
  ! CHECK:     %[[V_31:[0-9]+]] = fir.declare %[[V_30]] {uniq_name = "_QFEy3"} : (!fir.ref<bf16>) -> !fir.ref<bf16>
  ! CHECK:     %[[V_32:[0-9]+]] = fir.alloca f32 {bindc_name = "y8", uniq_name = "_QFEy8"}
  ! CHECK:     %[[V_33:[0-9]+]] = fir.declare %[[V_32]] {uniq_name = "_QFEy8"} : (!fir.ref<f32>) -> !fir.ref<f32>
  integer(2) n2
  integer(8) n8
  integer(16) n16
  real(2) x2, y2
  real(3) x3, y3

  ! CHECK:     fir.store %cst{{[_0-9]*}} to %[[V_19]] : !fir.ref<f32>
  x = -200.7

  ! CHECK:     %[[V_34:[0-9]+]] = fir.address_of(@_QQro._QM__fortran_builtinsT__builtin_ieee_round_type.0)
  ! CHECK:     %[[V_35:[0-9]+]] = fir.declare %[[V_34]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro._QM__fortran_builtinsT__builtin_ieee_round_type.0"}
  ! CHECK:     %[[V_36:[0-9]+]] = fir.load %[[V_19]] : !fir.ref<f32>
  ! CHECK:     %[[V_37:[0-9]+]] = fir.call @llvm.get.rounding() fastmath<contract> : () -> i32
  ! CHECK:     %[[V_38:[0-9]+]] = fir.field_index _QM__fortran_builtinsT__builtin_ieee_round_type.mode, !fir.type<_QM__fortran_builtinsT__builtin_ieee_round_type{_QM__fortran_builtinsT__builtin_ieee_round_type.mode:i8}>
  ! CHECK:     %[[V_39:[0-9]+]] = fir.coordinate_of %[[V_35]], %[[V_38]]
  ! CHECK:     %[[V_40:[0-9]+]] = fir.load %[[V_39]] : !fir.ref<i8>
  ! CHECK:     %[[V_41:[0-9]+]] = fir.convert %[[V_40]] : (i8) -> i32
  ! CHECK:     fir.call @llvm.set.rounding(%[[V_41]]) fastmath<contract> : (i32) -> ()
  ! CHECK:     %[[V_42:[0-9]+]] = fir.call @llvm.nearbyint.f32(%[[V_36]]) fastmath<contract> : (f32) -> f32
  ! CHECK:     fir.call @llvm.set.rounding(%[[V_37]]) fastmath<contract> : (i32) -> ()
  ! CHECK:     fir.store %[[V_42]] to %[[V_27]] : !fir.ref<f32>
  y = ieee_rint(x, ieee_nearest)

  ! CHECK:     %[[V_43:[0-9]+]] = fir.declare %[[V_34]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro._QM__fortran_builtinsT__builtin_ieee_round_type.0"}
  ! CHECK:     %[[V_44:[0-9]+]] = fir.load %[[V_19]] : !fir.ref<f32>
  ! CHECK:     %[[V_45:[0-9]+]] = fir.call @llvm.get.rounding() fastmath<contract> : () -> i32
  ! CHECK:     %[[V_46:[0-9]+]] = fir.coordinate_of %[[V_43]], %[[V_38]]
  ! CHECK:     %[[V_47:[0-9]+]] = fir.load %[[V_46]] : !fir.ref<i8>
  ! CHECK:     %[[V_48:[0-9]+]] = fir.convert %[[V_47]] : (i8) -> i32
  ! CHECK:     fir.call @llvm.set.rounding(%[[V_48]]) fastmath<contract> : (i32) -> ()
  ! CHECK:     %[[V_49:[0-9]+]] = fir.call @llvm.nearbyint.f32(%[[V_44]]) fastmath<contract> : (f32) -> f32
  ! CHECK:     fir.call @llvm.set.rounding(%[[V_45]]) fastmath<contract> : (i32) -> ()
  ! CHECK:     %[[V_50:[0-9]+]] = fir.convert %c-2147483648{{.*}} : (i32) -> f32
  ! CHECK:     %[[V_51:[0-9]+]] = arith.negf %[[V_50]] fastmath<contract> : f32
  ! CHECK:     %[[V_52:[0-9]+]] = arith.cmpf oge, %[[V_49]], %[[V_50]] fastmath<contract> : f32
  ! CHECK:     %[[V_53:[0-9]+]] = arith.cmpf olt, %[[V_49]], %[[V_51]] fastmath<contract> : f32
  ! CHECK:     %[[V_54:[0-9]+]] = arith.andi %[[V_52]], %[[V_53]] : i1
  ! CHECK:     %[[V_55:[0-9]+]] = fir.if %[[V_54]] -> (i32) {
  ! CHECK:       %[[V_128:[0-9]+]] = arith.cmpf one, %[[V_44]], %[[V_49]] fastmath<contract> : f32
  ! CHECK:       fir.if %[[V_128]] {
  ! CHECK:         %[[V_130:[0-9]+]] = fir.call @_FortranAMapException(%c32{{.*}}) fastmath<contract> : (i32) -> i32
  ! CHECK:         %[[V_131:[0-9]+]] = fir.call @feraiseexcept(%[[V_130]]) fastmath<contract> : (i32) -> i32
  ! CHECK:       }
  ! CHECK:       %[[V_129:[0-9]+]] = fir.convert %[[V_49]] : (f32) -> i32
  ! CHECK:       fir.result %[[V_129]] : i32
  ! CHECK:     } else {
  ! CHECK:       %[[V_128:[0-9]+]] = fir.call @_FortranAMapException(%c1{{.*}}) fastmath<contract> : (i32) -> i32
  ! CHECK:       %[[V_129:[0-9]+]] = fir.call @feraiseexcept(%[[V_128]]) fastmath<contract> : (i32) -> i32
  ! CHECK:       %[[V_130:[0-9]+]] = arith.select %[[V_52]], %c2147483647{{.*}}, %c-2147483648{{.*}} : i32
  ! CHECK:       fir.result %[[V_130]] : i32
  ! CHECK:     }
  ! CHECK:     fir.store %[[V_55]] to %[[V_11]] : !fir.ref<i32>
  n = ieee_int(x, ieee_nearest)
! print*, x, ' -> ', y, ' -> ', n

  ! CHECK:     fir.store %cst{{[_0-9]*}} to %[[V_21]] : !fir.ref<f16>
  x2 = huge(x2)

  ! CHECK:     %[[V_56:[0-9]+]] = fir.address_of(@_QQro._QM__fortran_builtinsT__builtin_ieee_round_type.1)
  ! CHECK:     %[[V_57:[0-9]+]] = fir.declare %[[V_56]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro._QM__fortran_builtinsT__builtin_ieee_round_type.1"}
  ! CHECK:     %[[V_58:[0-9]+]] = fir.load %[[V_21]] : !fir.ref<f16>
  ! CHECK:     %[[V_59:[0-9]+]] = fir.call @llvm.get.rounding() fastmath<contract> : () -> i32
  ! CHECK:     %[[V_60:[0-9]+]] = fir.coordinate_of %[[V_57]], %[[V_38]]
  ! CHECK:     %[[V_61:[0-9]+]] = fir.load %[[V_60]] : !fir.ref<i8>
  ! CHECK:     %[[V_62:[0-9]+]] = fir.convert %[[V_61]] : (i8) -> i32
  ! CHECK:     fir.call @llvm.set.rounding(%[[V_62]]) fastmath<contract> : (i32) -> ()
  ! CHECK:     %[[V_63:[0-9]+]] = fir.convert %[[V_58]] : (f16) -> f32
  ! CHECK:     %[[V_64:[0-9]+]] = fir.call @llvm.nearbyint.f32(%[[V_63]]) fastmath<contract> : (f32) -> f32
  ! CHECK:     %[[V_65:[0-9]+]] = fir.convert %[[V_64]] : (f32) -> f16
  ! CHECK:     fir.call @llvm.set.rounding(%[[V_59]]) fastmath<contract> : (i32) -> ()
  ! CHECK:     fir.store %[[V_65]] to %[[V_29]] : !fir.ref<f16>
  y2 = ieee_rint(x2, ieee_up)

  ! CHECK:     %[[V_66:[0-9]+]] = fir.declare %[[V_56]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro._QM__fortran_builtinsT__builtin_ieee_round_type.1"}
  ! CHECK:     %[[V_67:[0-9]+]] = fir.load %[[V_21]] : !fir.ref<f16>
  ! CHECK:     %[[V_68:[0-9]+]] = fir.call @llvm.get.rounding() fastmath<contract> : () -> i32
  ! CHECK:     %[[V_69:[0-9]+]] = fir.coordinate_of %[[V_66]], %[[V_38]]
  ! CHECK:     %[[V_70:[0-9]+]] = fir.load %[[V_69]] : !fir.ref<i8>
  ! CHECK:     %[[V_71:[0-9]+]] = fir.convert %[[V_70]] : (i8) -> i32
  ! CHECK:     fir.call @llvm.set.rounding(%[[V_71]]) fastmath<contract> : (i32) -> ()
  ! CHECK:     %[[V_72:[0-9]+]] = fir.convert %[[V_67]] : (f16) -> f32
  ! CHECK:     %[[V_73:[0-9]+]] = fir.call @llvm.nearbyint.f32(%[[V_72]]) fastmath<contract> : (f32) -> f32
  ! CHECK:     %[[V_74:[0-9]+]] = fir.convert %[[V_73]] : (f32) -> f16
  ! CHECK:     fir.call @llvm.set.rounding(%[[V_68]]) fastmath<contract> : (i32) -> ()
  ! CHECK:     %[[V_75:[0-9]+]] = fir.convert %c-9223372036854775808{{.*}} : (i64) -> f16
  ! CHECK:     %[[V_76:[0-9]+]] = arith.negf %[[V_75]] fastmath<contract> : f16
  ! CHECK:     %[[V_77:[0-9]+]] = arith.cmpf oge, %[[V_74]], %[[V_75]] fastmath<contract> : f16
  ! CHECK:     %[[V_78:[0-9]+]] = arith.cmpf olt, %[[V_74]], %[[V_76]] fastmath<contract> : f16
  ! CHECK:     %[[V_79:[0-9]+]] = arith.andi %[[V_77]], %[[V_78]] : i1
  ! CHECK:     %[[V_80:[0-9]+]] = fir.if %[[V_79]] -> (i64) {
  ! CHECK:       %[[V_128:[0-9]+]] = arith.cmpf one, %[[V_67]], %[[V_74]] fastmath<contract> : f16
  ! CHECK:       fir.if %[[V_128]] {
  ! CHECK:         %[[V_130:[0-9]+]] = fir.call @_FortranAMapException(%c32{{.*}}) fastmath<contract> : (i32) -> i32
  ! CHECK:         %[[V_131:[0-9]+]] = fir.call @feraiseexcept(%[[V_130]]) fastmath<contract> : (i32) -> i32
  ! CHECK:       }
  ! CHECK:       %[[V_129:[0-9]+]] = fir.convert %[[V_74]] : (f16) -> i64
  ! CHECK:       fir.result %[[V_129]] : i64
  ! CHECK:     } else {
  ! CHECK:       %[[V_128:[0-9]+]] = fir.call @_FortranAMapException(%c1{{.*}}) fastmath<contract> : (i32) -> i32
  ! CHECK:       %[[V_129:[0-9]+]] = fir.call @feraiseexcept(%[[V_128]]) fastmath<contract> : (i32) -> i32
  ! CHECK:       %[[V_130:[0-9]+]] = arith.select %[[V_77]], %c9223372036854775807{{.*}}, %c-9223372036854775808{{.*}} : i64
  ! CHECK:       fir.result %[[V_130]] : i64
  ! CHECK:     }
  ! CHECK:     fir.store %[[V_80]] to %[[V_17]] : !fir.ref<i64>
  n8 = ieee_int(x2, ieee_up, 8)

! print*, x2, ' -> ', y2, ' -> ', n8

  ! CHECK:     fir.store %cst{{[_0-9]*}} to %[[V_23]] : !fir.ref<bf16>
  x3 = -0.

  ! CHECK:     %[[V_81:[0-9]+]] = fir.load %[[V_23]] : !fir.ref<bf16>
  ! CHECK:     %[[V_82:[0-9]+]] = fir.convert %[[V_81]] : (bf16) -> f32
  ! CHECK:     %[[V_83:[0-9]+]] = fir.call @llvm.nearbyint.f32(%[[V_82]]) fastmath<contract> : (f32) -> f32
  ! CHECK:     %[[V_84:[0-9]+]] = fir.convert %[[V_83]] : (f32) -> bf16
  ! CHECK:     %[[V_85:[0-9]+]] = arith.cmpf one, %[[V_81]], %[[V_84]] fastmath<contract> : bf16
  ! CHECK:     fir.if %[[V_85]] {
  ! CHECK:       %[[V_128:[0-9]+]] = fir.call @_FortranAMapException(%c32{{.*}}) fastmath<contract> : (i32) -> i32
  ! CHECK:       %[[V_129:[0-9]+]] = fir.call @feraiseexcept(%[[V_128]]) fastmath<contract> : (i32) -> i32
  ! CHECK:     }
  ! CHECK:     fir.store %[[V_84]] to %[[V_31]] : !fir.ref<bf16>
  y3 = ieee_rint(x3)

  ! CHECK:     %[[V_86:[0-9]+]] = fir.address_of(@_QQro._QM__fortran_builtinsT__builtin_ieee_round_type.2)
  ! CHECK:     %[[V_87:[0-9]+]] = fir.declare %[[V_86]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro._QM__fortran_builtinsT__builtin_ieee_round_type.2"}
  ! CHECK:     %[[V_88:[0-9]+]] = fir.load %[[V_23]] : !fir.ref<bf16>
  ! CHECK:     %[[V_89:[0-9]+]] = fir.call @llvm.get.rounding() fastmath<contract> : () -> i32
  ! CHECK:     %[[V_90:[0-9]+]] = fir.coordinate_of %[[V_87]], %[[V_38]]
  ! CHECK:     %[[V_91:[0-9]+]] = fir.load %[[V_90]] : !fir.ref<i8>
  ! CHECK:     %[[V_92:[0-9]+]] = fir.convert %[[V_91]] : (i8) -> i32
  ! CHECK:     fir.call @llvm.set.rounding(%[[V_92]]) fastmath<contract> : (i32) -> ()
  ! CHECK:     %[[V_93:[0-9]+]] = fir.convert %[[V_88]] : (bf16) -> f32
  ! CHECK:     %[[V_94:[0-9]+]] = fir.call @llvm.nearbyint.f32(%[[V_93]]) fastmath<contract> : (f32) -> f32
  ! CHECK:     %[[V_95:[0-9]+]] = fir.convert %[[V_94]] : (f32) -> bf16
  ! CHECK:     fir.call @llvm.set.rounding(%[[V_89]]) fastmath<contract> : (i32) -> ()
  ! CHECK:     %[[V_96:[0-9]+]] = fir.convert %c-170141183460469231731687303715884105728{{.*}} : (i128) -> bf16
  ! CHECK:     %[[V_97:[0-9]+]] = arith.negf %[[V_96]] fastmath<contract> : bf16
  ! CHECK:     %[[V_98:[0-9]+]] = arith.cmpf oge, %[[V_95]], %[[V_96]] fastmath<contract> : bf16
  ! CHECK:     %[[V_99:[0-9]+]] = arith.cmpf olt, %[[V_95]], %[[V_97]] fastmath<contract> : bf16
  ! CHECK:     %[[V_100:[0-9]+]] = arith.andi %[[V_98]], %[[V_99]] : i1
  ! CHECK:     %[[V_101:[0-9]+]] = fir.if %[[V_100]] -> (i128) {
  ! CHECK:       %[[V_128:[0-9]+]] = arith.cmpf one, %[[V_88]], %[[V_95]] fastmath<contract> : bf16
  ! CHECK:       fir.if %[[V_128]] {
  ! CHECK:         %[[V_130:[0-9]+]] = fir.call @_FortranAMapException(%c32{{.*}}) fastmath<contract> : (i32) -> i32
  ! CHECK:         %[[V_131:[0-9]+]] = fir.call @feraiseexcept(%[[V_130]]) fastmath<contract> : (i32) -> i32
  ! CHECK:       }
  ! CHECK:       %[[V_129:[0-9]+]] = fir.convert %[[V_95]] : (bf16) -> i128
  ! CHECK:       fir.result %[[V_129]] : i128
  ! CHECK:     } else {
  ! CHECK:       %[[V_128:[0-9]+]] = fir.call @_FortranAMapException(%c1{{.*}}) fastmath<contract> : (i32) -> i32
  ! CHECK:       %[[V_129:[0-9]+]] = fir.call @feraiseexcept(%[[V_128]]) fastmath<contract> : (i32) -> i32
  ! CHECK:       %[[V_130:[0-9]+]] = arith.select %[[V_98]], %c170141183460469231731687303715884105727{{.*}}, %c-170141183460469231731687303715884105728{{.*}} : i128
  ! CHECK:       fir.result %[[V_130]] : i128
  ! CHECK:     }
  ! CHECK:     fir.store %[[V_101]] to %[[V_13]] : !fir.ref<i128>
  n16 = ieee_int(x3, ieee_away, 16)

! print*, x3, ' -> ', y3, ' -> ', n16

  ! CHECK:     %[[V_102:[0-9]+]] = fir.address_of(@_QQro._QMieee_arithmeticTieee_class_type.3)
  ! CHECK:     %[[V_103:[0-9]+]] = fir.declare %[[V_102]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro._QMieee_arithmeticTieee_class_type.3"}
  ! CHECK:     %[[V_104:[0-9]+]] = fir.field_index _QMieee_arithmeticTieee_class_type.which, !fir.type<_QMieee_arithmeticTieee_class_type{_QMieee_arithmeticTieee_class_type.which:i8}>
  ! CHECK:     %[[V_105:[0-9]+]] = fir.coordinate_of %[[V_103]], %[[V_104]]
  ! CHECK:     %[[V_106:[0-9]+]] = fir.load %[[V_105]] : !fir.ref<i8>
  ! CHECK:     %[[V_107:[0-9]+]] = fir.address_of(@_FortranAIeeeValueTable_4) : !fir.ref<!fir.array<12xi32>>
  ! CHECK:     %[[V_108:[0-9]+]] = fir.coordinate_of %[[V_107]], %[[V_106]]
  ! CHECK:     %[[V_109:[0-9]+]] = fir.load %[[V_108]] : !fir.ref<i32>
  ! CHECK:     %[[V_110:[0-9]+]] = arith.bitcast %[[V_109]] : i32 to f32
  ! CHECK:     fir.store %[[V_110]] to %[[V_25]] : !fir.ref<f32>
  x8 = ieee_value(x8, ieee_positive_inf)

  ! CHECK:     %[[V_111:[0-9]+]] = fir.load %[[V_25]] : !fir.ref<f32>
  ! CHECK:     %[[V_112:[0-9]+]] = fir.call @llvm.nearbyint.f32(%[[V_111]]) fastmath<contract> : (f32) -> f32
  ! CHECK:     %[[V_113:[0-9]+]] = arith.cmpf one, %[[V_111]], %[[V_112]] fastmath<contract> : f32
  ! CHECK:     fir.if %[[V_113]] {
  ! CHECK:       %[[V_128:[0-9]+]] = fir.call @_FortranAMapException(%c32{{.*}}) fastmath<contract> : (i32) -> i32
  ! CHECK:       %[[V_129:[0-9]+]] = fir.call @feraiseexcept(%[[V_128]]) fastmath<contract> : (i32) -> i32
  ! CHECK:     }
  ! CHECK:     fir.store %[[V_112]] to %[[V_33]] : !fir.ref<f32>
  y8 = ieee_rint(x8)

  ! CHECK:     %[[V_114:[0-9]+]] = fir.address_of(@_QQro._QM__fortran_builtinsT__builtin_ieee_round_type.4)
  ! CHECK:     %[[V_115:[0-9]+]] = fir.declare %[[V_114]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro._QM__fortran_builtinsT__builtin_ieee_round_type.4"}
  ! CHECK:     %[[V_116:[0-9]+]] = fir.load %[[V_25]] : !fir.ref<f32>
  ! CHECK:     %[[V_117:[0-9]+]] = fir.call @llvm.get.rounding() fastmath<contract> : () -> i32
  ! CHECK:     %[[V_118:[0-9]+]] = fir.coordinate_of %[[V_115]], %[[V_38]]
  ! CHECK:     %[[V_119:[0-9]+]] = fir.load %[[V_118]] : !fir.ref<i8>
  ! CHECK:     %[[V_120:[0-9]+]] = fir.convert %[[V_119]] : (i8) -> i32
  ! CHECK:     fir.call @llvm.set.rounding(%[[V_120]]) fastmath<contract> : (i32) -> ()
  ! CHECK:     %[[V_121:[0-9]+]] = fir.call @llvm.nearbyint.f32(%[[V_116]]) fastmath<contract> : (f32) -> f32
  ! CHECK:     fir.call @llvm.set.rounding(%[[V_117]]) fastmath<contract> : (i32) -> ()
  ! CHECK:     %[[V_122:[0-9]+]] = fir.convert %c-32768{{.*}} : (i16) -> f32
  ! CHECK:     %[[V_123:[0-9]+]] = arith.negf %[[V_122]] fastmath<contract> : f32
  ! CHECK:     %[[V_124:[0-9]+]] = arith.cmpf oge, %[[V_121]], %[[V_122]] fastmath<contract> : f32
  ! CHECK:     %[[V_125:[0-9]+]] = arith.cmpf olt, %[[V_121]], %[[V_123]] fastmath<contract> : f32
  ! CHECK:     %[[V_126:[0-9]+]] = arith.andi %[[V_124]], %[[V_125]] : i1
  ! CHECK:     %[[V_127:[0-9]+]] = fir.if %[[V_126]] -> (i16) {
  ! CHECK:       %[[V_128:[0-9]+]] = arith.cmpf one, %[[V_116]], %[[V_121]] fastmath<contract> : f32
  ! CHECK:       fir.if %[[V_128]] {
  ! CHECK:         %[[V_130:[0-9]+]] = fir.call @_FortranAMapException(%c32{{.*}}) fastmath<contract> : (i32) -> i32
  ! CHECK:         %[[V_131:[0-9]+]] = fir.call @feraiseexcept(%[[V_130]]) fastmath<contract> : (i32) -> i32
  ! CHECK:       }
  ! CHECK:       %[[V_129:[0-9]+]] = fir.convert %[[V_121]] : (f32) -> i16
  ! CHECK:       fir.result %[[V_129]] : i16
  ! CHECK:     } else {
  ! CHECK:       %[[V_128:[0-9]+]] = fir.call @_FortranAMapException(%c1{{.*}}) fastmath<contract> : (i32) -> i32
  ! CHECK:       %[[V_129:[0-9]+]] = fir.call @feraiseexcept(%[[V_128]]) fastmath<contract> : (i32) -> i32
  ! CHECK:       %[[V_130:[0-9]+]] = arith.select %[[V_124]], %c32767{{.*}}, %c-32768{{.*}} : i16
  ! CHECK:       fir.result %[[V_130]] : i16
  ! CHECK:     }
  ! CHECK:     fir.store %[[V_127]] to %[[V_15]] : !fir.ref<i16>
  n2 = ieee_int(x8, ieee_to_zero, 2)

! print*, x8, ' -> ', y8, ' -> ', n2
end
