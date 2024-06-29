! RUN: bbc -emit-fir -o - %s | FileCheck %s

function tag(x)
  use ieee_arithmetic
  character(12) :: tag
  real(4) :: x
  tag = '?????'
  if (ieee_class(x) == ieee_signaling_nan    ) tag = 'snan'
  if (ieee_class(x) == ieee_quiet_nan        ) tag = 'qnan'
  if (ieee_class(x) == ieee_negative_inf     ) tag = 'neg_inf'
  if (ieee_class(x) == ieee_negative_normal  ) tag = 'neg_norm'
  if (ieee_class(x) == ieee_negative_denormal) tag = 'neg_denorm'
  if (ieee_class(x) == ieee_negative_zero    ) tag = 'neg_zero'
  if (ieee_class(x) == ieee_positive_zero    ) tag = 'pos_zero'
  if (ieee_class(x) == ieee_positive_denormal) tag = 'pos_denorm'
  if (ieee_class(x) == ieee_positive_normal  ) tag = 'pos_norm'
  if (ieee_class(x) == ieee_positive_inf     ) tag = 'pos_inf'
end

! CHECK-LABEL: c.func @_QQmain
program p
  use ieee_arithmetic
  character(12) :: tag

  ! CHECK:     %[[V_16:[0-9]+]] = fir.alloca f32 {bindc_name = "a", uniq_name = "_QFEa"}
  ! CHECK:     %[[V_17:[0-9]+]] = fir.declare %[[V_16]] {uniq_name = "_QFEa"} : (!fir.ref<f32>) -> !fir.ref<f32>
  ! CHECK:     %[[V_18:[0-9]+]] = fir.alloca f32 {bindc_name = "b", uniq_name = "_QFEb"}
  ! CHECK:     %[[V_19:[0-9]+]] = fir.declare %[[V_18]] {uniq_name = "_QFEb"} : (!fir.ref<f32>) -> !fir.ref<f32>
  ! CHECK:     %[[V_20:[0-9]+]] = fir.alloca !fir.logical<4> {bindc_name = "flag_value", uniq_name = "_QFEflag_value"}
  ! CHECK:     %[[V_21:[0-9]+]] = fir.declare %[[V_20]] {uniq_name = "_QFEflag_value"} : (!fir.ref<!fir.logical<4>>) -> !fir.ref<!fir.logical<4>>
  ! CHECK:     %[[V_82:[0-9]+]] = fir.alloca f32 {bindc_name = "r", uniq_name = "_QFEr"}
  ! CHECK:     %[[V_83:[0-9]+]] = fir.declare %[[V_82]] {uniq_name = "_QFEr"} : (!fir.ref<f32>) -> !fir.ref<f32>
  logical :: flag_value
  real(4) :: x(22), a, b, r

  ! CHECK:     %[[V_92:[0-9]+]] = fir.address_of(@_FortranAIeeeValueTable_4) : !fir.ref<!fir.array<12xi32>>

  x( 1) = ieee_value(a, ieee_signaling_nan)
  x( 2) = ieee_value(a, ieee_quiet_nan)
  x( 3) = ieee_value(a, ieee_negative_inf)
  x( 4) = -huge(a)
  x( 5) = -1000
  x( 6) = -10
  x( 7) = ieee_value(a, ieee_negative_normal)
  x( 8) = -.1
  x( 9) = -.001
  x(10) = -tiny(a)
  x(11) = ieee_value(a, ieee_negative_denormal)
  x(12) = ieee_value(a, ieee_negative_zero)
  x(13) = ieee_value(a, ieee_positive_zero)
  x(14) = ieee_value(a, ieee_positive_denormal)
  x(15) = tiny(a)
  x(16) = .001
  x(17) = .1
  x(18) = ieee_value(a, ieee_positive_normal)
  x(19) = 10
  x(20) = 1000
  x(21) = huge(a)
  x(22) = ieee_value(a, ieee_positive_inf)

  4 format(A8,'(',f10.2,z9.8  ,', ',f10.2,z9.8  ,') = ',f10.2,L3,'  ',A)

  do i = lbound(x,1), ubound(x,1)
    print*
    do j = lbound(x,1), ubound(x,1)
      print*
      a = x(i)
      b = x(j)

      ! CHECK:     %[[V_201:[0-9]+]] = fir.address_of(@_QQro._QM__fortran_builtinsT__builtin_ieee_flag_type.10) : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>
      ! CHECK:     %[[V_202:[0-9]+]] = fir.declare %[[V_201]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro._QM__fortran_builtinsT__builtin_ieee_flag_type.10"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>
      ! CHECK:     %[[V_203:[0-9]+]] = fir.field_index _QM__fortran_builtinsT__builtin_ieee_flag_type.flag, !fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>
      ! CHECK:     %[[V_204:[0-9]+]] = fir.coordinate_of %[[V_202]], %[[V_203]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>, !fir.field) -> !fir.ref<i8>
      ! CHECK:     %[[V_205:[0-9]+]] = fir.load %[[V_204]] : !fir.ref<i8>
      ! CHECK:     %[[V_206:[0-9]+]] = fir.convert %[[V_205]] : (i8) -> i32
      ! CHECK:     %[[V_207:[0-9]+]] = fir.call @_FortranAMapException(%[[V_206]]) fastmath<contract> : (i32) -> i32
      ! CHECK:     fir.if %false{{[_0-9]*}} {
      ! CHECK:       %[[V_692:[0-9]+]] = fir.call @feraiseexcept(%[[V_207]]) fastmath<contract> : (i32) -> i32
      ! CHECK:     } else {
      ! CHECK:       %[[V_692:[0-9]+]] = fir.call @feclearexcept(%[[V_207]]) fastmath<contract> : (i32) -> i32
      ! CHECK:     }
      ! CHECK:     %[[V_208:[0-9]+]] = fir.load %[[V_17]] : !fir.ref<f32>
      ! CHECK:     %[[V_209:[0-9]+]] = fir.load %[[V_19]] : !fir.ref<f32>
      ! CHECK:     %[[V_210:[0-9]+]] = arith.cmpf olt, %[[V_208]], %[[V_209]] {{.*}} : f32
      ! CHECK:     %[[V_211:[0-9]+]] = fir.if %[[V_210]] -> (f32) {
      ! CHECK:       fir.result %[[V_209]] : f32
      ! CHECK:     } else {
      ! CHECK:       %[[V_692:[0-9]+]] = arith.cmpf ogt, %[[V_208]], %[[V_209]] {{.*}} : f32
      ! CHECK:       %[[V_693:[0-9]+]] = fir.if %[[V_692]] -> (f32) {
      ! CHECK:         fir.result %[[V_208]] : f32
      ! CHECK:       } else {
      ! CHECK:         %[[V_694:[0-9]+]] = arith.cmpf oeq, %[[V_208]], %[[V_209]] {{.*}} : f32
      ! CHECK:         %[[V_695:[0-9]+]] = fir.if %[[V_694]] -> (f32) {
      ! CHECK:           %[[V_696:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_208]]) <{bit = 960 : i32}> : (f32) -> i1
      ! CHECK:           %[[V_697:[0-9]+]] = arith.select %[[V_696]], %[[V_208]], %[[V_209]] : f32
      ! CHECK:           fir.result %[[V_697]] : f32
      ! CHECK:         } else {
      ! CHECK:           %[[V_696:[0-9]+]] = fir.coordinate_of %[[V_92]], %c2{{.*}} : (!fir.ref<!fir.array<12xi32>>, i8) -> !fir.ref<i32>
      ! CHECK:           %[[V_697:[0-9]+]] = fir.load %[[V_696]] : !fir.ref<i32>
      ! CHECK:           %[[V_698:[0-9]+]] = arith.bitcast %[[V_697]] : i32 to f32
      ! CHECK-DAG:       %[[V_699:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_209]]) <{bit = 1 : i32}> : (f32) -> i1
      ! CHECK-DAG:       %[[V_700:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_208]]) <{bit = 1 : i32}> : (f32) -> i1
      ! CHECK:           %[[V_701:[0-9]+]] = arith.ori %[[V_700]], %[[V_699]] : i1
      ! CHECK:           fir.if %[[V_701]] {
      ! CHECK:             %[[V_702:[0-9]+]] = fir.call @_FortranAMapException(%c1{{.*}}) fastmath<contract> : (i32) -> i32
      ! CHECK:             %[[V_703:[0-9]+]] = fir.call @feraiseexcept(%[[V_702]]) fastmath<contract> : (i32) -> i32
      ! CHECK:           }
      ! CHECK:           fir.result %[[V_698]] : f32
      ! CHECK:         }
      ! CHECK:         fir.result %[[V_695]] : f32
      ! CHECK:       }
      ! CHECK:       fir.result %[[V_693]] : f32
      ! CHECK:     }
      ! CHECK:     fir.store %[[V_211]] to %[[V_83]] : !fir.ref<f32>
      ! CHECK:     %[[V_212:[0-9]+]] = fir.declare %[[V_201]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro._QM__fortran_builtinsT__builtin_ieee_flag_type.10"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>
      ! CHECK:     %[[V_213:[0-9]+]] = fir.coordinate_of %[[V_212]], %[[V_203]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>, !fir.field) -> !fir.ref<i8>
      ! CHECK:     %[[V_214:[0-9]+]] = fir.load %[[V_213]] : !fir.ref<i8>
      ! CHECK:     %[[V_215:[0-9]+]] = fir.convert %[[V_214]] : (i8) -> i32
      ! CHECK:     %[[V_216:[0-9]+]] = fir.call @_FortranAMapException(%[[V_215]]) fastmath<contract> : (i32) -> i32
      ! CHECK:     %[[V_217:[0-9]+]] = fir.call @fetestexcept(%[[V_216]]) fastmath<contract> : (i32) -> i32
      ! CHECK:     %[[V_218:[0-9]+]] = arith.cmpi ne, %[[V_217]], %c0{{.*}} : i32
      ! CHECK:     %[[V_219:[0-9]+]] = fir.convert %[[V_218]] : (i1) -> !fir.logical<4>
      ! CHECK:     fir.store %[[V_219]] to %[[V_21]] : !fir.ref<!fir.logical<4>>
      call ieee_set_flag(ieee_invalid, .false.)
      r = ieee_max(a, b)
      call ieee_get_flag(ieee_invalid, flag_value)
      write(*, 4) 'max    ', a, a, b, b, r, flag_value, trim(tag(r))

      ! CHECK:     %[[V_268:[0-9]+]] = fir.declare %[[V_201]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro._QM__fortran_builtinsT__builtin_ieee_flag_type.10"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>
      ! CHECK:     %[[V_269:[0-9]+]] = fir.coordinate_of %[[V_268]], %[[V_203]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>, !fir.field) -> !fir.ref<i8>
      ! CHECK:     %[[V_270:[0-9]+]] = fir.load %[[V_269]] : !fir.ref<i8>
      ! CHECK:     %[[V_271:[0-9]+]] = fir.convert %[[V_270]] : (i8) -> i32
      ! CHECK:     %[[V_272:[0-9]+]] = fir.call @_FortranAMapException(%[[V_271]]) fastmath<contract> : (i32) -> i32
      ! CHECK:     fir.if %false{{[_0-9]*}} {
      ! CHECK:       %[[V_692:[0-9]+]] = fir.call @feraiseexcept(%[[V_272]]) fastmath<contract> : (i32) -> i32
      ! CHECK:     } else {
      ! CHECK:       %[[V_692:[0-9]+]] = fir.call @feclearexcept(%[[V_272]]) fastmath<contract> : (i32) -> i32
      ! CHECK:     }
      ! CHECK:     %[[V_273:[0-9]+]] = fir.load %[[V_17]] : !fir.ref<f32>
      ! CHECK:     %[[V_274:[0-9]+]] = fir.load %[[V_19]] : !fir.ref<f32>
      ! CHECK:     %[[V_275:[0-9]+]] = math.copysign %[[V_273]], %cst{{[_0-9]*}} fastmath<contract> : f32
      ! CHECK:     %[[V_276:[0-9]+]] = math.copysign %[[V_274]], %cst{{[_0-9]*}} fastmath<contract> : f32
      ! CHECK:     %[[V_277:[0-9]+]] = arith.cmpf olt, %[[V_275]], %[[V_276]] {{.*}} : f32
      ! CHECK:     %[[V_278:[0-9]+]] = fir.if %[[V_277]] -> (f32) {
      ! CHECK:       fir.result %[[V_274]] : f32
      ! CHECK:     } else {
      ! CHECK:       %[[V_692:[0-9]+]] = arith.cmpf ogt, %[[V_275]], %[[V_276]] {{.*}} : f32
      ! CHECK:       %[[V_693:[0-9]+]] = fir.if %[[V_692]] -> (f32) {
      ! CHECK:         fir.result %[[V_273]] : f32
      ! CHECK:       } else {
      ! CHECK:         %[[V_694:[0-9]+]] = arith.cmpf oeq, %[[V_275]], %[[V_276]] {{.*}} : f32
      ! CHECK:         %[[V_695:[0-9]+]] = fir.if %[[V_694]] -> (f32) {
      ! CHECK:           %[[V_696:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_273]]) <{bit = 960 : i32}> : (f32) -> i1
      ! CHECK:           %[[V_697:[0-9]+]] = arith.select %[[V_696]], %[[V_273]], %[[V_274]] : f32
      ! CHECK:           fir.result %[[V_697]] : f32
      ! CHECK:         } else {
      ! CHECK:           %[[V_696:[0-9]+]] = fir.coordinate_of %[[V_92]], %c2{{.*}} : (!fir.ref<!fir.array<12xi32>>, i8) -> !fir.ref<i32>
      ! CHECK:           %[[V_697:[0-9]+]] = fir.load %[[V_696]] : !fir.ref<i32>
      ! CHECK:           %[[V_698:[0-9]+]] = arith.bitcast %[[V_697]] : i32 to f32
      ! CHECK-DAG:       %[[V_699:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_274]]) <{bit = 1 : i32}> : (f32) -> i1
      ! CHECK-DAG:       %[[V_700:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_273]]) <{bit = 1 : i32}> : (f32) -> i1
      ! CHECK:           %[[V_701:[0-9]+]] = arith.ori %[[V_700]], %[[V_699]] : i1
      ! CHECK:           fir.if %[[V_701]] {
      ! CHECK:             %[[V_702:[0-9]+]] = fir.call @_FortranAMapException(%c1{{.*}}) fastmath<contract> : (i32) -> i32
      ! CHECK:             %[[V_703:[0-9]+]] = fir.call @feraiseexcept(%[[V_702]]) fastmath<contract> : (i32) -> i32
      ! CHECK:           }
      ! CHECK:           fir.result %[[V_698]] : f32
      ! CHECK:         }
      ! CHECK:         fir.result %[[V_695]] : f32
      ! CHECK:       }
      ! CHECK:       fir.result %[[V_693]] : f32
      ! CHECK:     }
      ! CHECK:     fir.store %[[V_278]] to %[[V_83]] : !fir.ref<f32>
      ! CHECK:     %[[V_279:[0-9]+]] = fir.declare %[[V_201]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro._QM__fortran_builtinsT__builtin_ieee_flag_type.10"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>
      ! CHECK:     %[[V_280:[0-9]+]] = fir.coordinate_of %[[V_279]], %[[V_203]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>, !fir.field) -> !fir.ref<i8>
      ! CHECK:     %[[V_281:[0-9]+]] = fir.load %[[V_280]] : !fir.ref<i8>
      ! CHECK:     %[[V_282:[0-9]+]] = fir.convert %[[V_281]] : (i8) -> i32
      ! CHECK:     %[[V_283:[0-9]+]] = fir.call @_FortranAMapException(%[[V_282]]) fastmath<contract> : (i32) -> i32
      ! CHECK:     %[[V_284:[0-9]+]] = fir.call @fetestexcept(%[[V_283]]) fastmath<contract> : (i32) -> i32
      ! CHECK:     %[[V_285:[0-9]+]] = arith.cmpi ne, %[[V_284]], %c0{{.*}} : i32
      ! CHECK:     %[[V_286:[0-9]+]] = fir.convert %[[V_285]] : (i1) -> !fir.logical<4>
      ! CHECK:     fir.store %[[V_286]] to %[[V_21]] : !fir.ref<!fir.logical<4>>
      call ieee_set_flag(ieee_invalid, .false.)
      r = ieee_max_mag(a, b)
      call ieee_get_flag(ieee_invalid, flag_value)
      write(*, 4) 'mag    ', a, a, b, b, r, flag_value, trim(tag(r))

      ! CHECK:     %[[V_329:[0-9]+]] = fir.declare %[[V_201]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro._QM__fortran_builtinsT__builtin_ieee_flag_type.10"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>
      ! CHECK:     %[[V_330:[0-9]+]] = fir.coordinate_of %[[V_329]], %[[V_203]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>, !fir.field) -> !fir.ref<i8>
      ! CHECK:     %[[V_331:[0-9]+]] = fir.load %[[V_330]] : !fir.ref<i8>
      ! CHECK:     %[[V_332:[0-9]+]] = fir.convert %[[V_331]] : (i8) -> i32
      ! CHECK:     %[[V_333:[0-9]+]] = fir.call @_FortranAMapException(%[[V_332]]) fastmath<contract> : (i32) -> i32
      ! CHECK:     fir.if %false{{[_0-9]*}} {
      ! CHECK:       %[[V_692:[0-9]+]] = fir.call @feraiseexcept(%[[V_333]]) fastmath<contract> : (i32) -> i32
      ! CHECK:     } else {
      ! CHECK:       %[[V_692:[0-9]+]] = fir.call @feclearexcept(%[[V_333]]) fastmath<contract> : (i32) -> i32
      ! CHECK:     }
      ! CHECK:     %[[V_334:[0-9]+]] = fir.load %[[V_17]] : !fir.ref<f32>
      ! CHECK:     %[[V_335:[0-9]+]] = fir.load %[[V_19]] : !fir.ref<f32>
      ! CHECK:     %[[V_336:[0-9]+]] = arith.cmpf olt, %[[V_334]], %[[V_335]] {{.*}} : f32
      ! CHECK:     %[[V_337:[0-9]+]] = fir.if %[[V_336]] -> (f32) {
      ! CHECK:       fir.result %[[V_335]] : f32
      ! CHECK:     } else {
      ! CHECK:       %[[V_692:[0-9]+]] = arith.cmpf ogt, %[[V_334]], %[[V_335]] {{.*}} : f32
      ! CHECK:       %[[V_693:[0-9]+]] = fir.if %[[V_692]] -> (f32) {
      ! CHECK:         fir.result %[[V_334]] : f32
      ! CHECK:       } else {
      ! CHECK:         %[[V_694:[0-9]+]] = arith.cmpf oeq, %[[V_334]], %[[V_335]] {{.*}} : f32
      ! CHECK:         %[[V_695:[0-9]+]] = fir.if %[[V_694]] -> (f32) {
      ! CHECK:           %[[V_696:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_334]]) <{bit = 960 : i32}> : (f32) -> i1
      ! CHECK:           %[[V_697:[0-9]+]] = arith.select %[[V_696]], %[[V_334]], %[[V_335]] : f32
      ! CHECK:           fir.result %[[V_697]] : f32
      ! CHECK:         } else {
      ! CHECK:           %[[V_696:[0-9]+]] = arith.cmpf ord, %[[V_334]], %[[V_334]] {{.*}} : f32
      ! CHECK:           %[[V_697:[0-9]+]] = arith.cmpf ord, %[[V_335]], %[[V_335]] {{.*}} : f32
      ! CHECK:           %[[V_698:[0-9]+]] = fir.coordinate_of %[[V_92]], %c2{{.*}} : (!fir.ref<!fir.array<12xi32>>, i8) -> !fir.ref<i32>
      ! CHECK:           %[[V_699:[0-9]+]] = fir.load %[[V_698]] : !fir.ref<i32>
      ! CHECK:           %[[V_700:[0-9]+]] = arith.bitcast %[[V_699]] : i32 to f32
      ! CHECK:           %[[V_701:[0-9]+]] = arith.select %[[V_697]], %[[V_335]], %[[V_700]] : f32
      ! CHECK:           %[[V_702:[0-9]+]] = arith.select %[[V_696]], %[[V_334]], %[[V_701]] : f32
      ! CHECK-DAG:       %[[V_703:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_335]]) <{bit = 1 : i32}> : (f32) -> i1
      ! CHECK-DAG:       %[[V_704:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_334]]) <{bit = 1 : i32}> : (f32) -> i1
      ! CHECK:           %[[V_705:[0-9]+]] = arith.ori %[[V_704]], %[[V_703]] : i1
      ! CHECK:           fir.if %[[V_705]] {
      ! CHECK:             %[[V_706:[0-9]+]] = fir.call @_FortranAMapException(%c1{{.*}}) fastmath<contract> : (i32) -> i32
      ! CHECK:             %[[V_707:[0-9]+]] = fir.call @feraiseexcept(%[[V_706]]) fastmath<contract> : (i32) -> i32
      ! CHECK:           }
      ! CHECK:           fir.result %[[V_702]] : f32
      ! CHECK:         }
      ! CHECK:         fir.result %[[V_695]] : f32
      ! CHECK:       }
      ! CHECK:       fir.result %[[V_693]] : f32
      ! CHECK:     }
      ! CHECK:     fir.store %[[V_337]] to %[[V_83]] : !fir.ref<f32>
      ! CHECK:     %[[V_338:[0-9]+]] = fir.declare %[[V_201]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro._QM__fortran_builtinsT__builtin_ieee_flag_type.10"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>
      ! CHECK:     %[[V_339:[0-9]+]] = fir.coordinate_of %[[V_338]], %[[V_203]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>, !fir.field) -> !fir.ref<i8>
      ! CHECK:     %[[V_340:[0-9]+]] = fir.load %[[V_339]] : !fir.ref<i8>
      ! CHECK:     %[[V_341:[0-9]+]] = fir.convert %[[V_340]] : (i8) -> i32
      ! CHECK:     %[[V_342:[0-9]+]] = fir.call @_FortranAMapException(%[[V_341]]) fastmath<contract> : (i32) -> i32
      ! CHECK:     %[[V_343:[0-9]+]] = fir.call @fetestexcept(%[[V_342]]) fastmath<contract> : (i32) -> i32
      ! CHECK:     %[[V_344:[0-9]+]] = arith.cmpi ne, %[[V_343]], %c0{{.*}} : i32
      ! CHECK:     %[[V_345:[0-9]+]] = fir.convert %[[V_344]] : (i1) -> !fir.logical<4>
      ! CHECK:     fir.store %[[V_345]] to %[[V_21]] : !fir.ref<!fir.logical<4>>
      call ieee_set_flag(ieee_invalid, .false.)
      r = ieee_max_num(a, b)
      call ieee_get_flag(ieee_invalid, flag_value)
      write(*, 4) 'max_num', a, a, b, b, r, flag_value, trim(tag(r))

      ! CHECK:     %[[V_388:[0-9]+]] = fir.declare %[[V_201]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro._QM__fortran_builtinsT__builtin_ieee_flag_type.10"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>
      ! CHECK:     %[[V_389:[0-9]+]] = fir.coordinate_of %[[V_388]], %[[V_203]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>, !fir.field) -> !fir.ref<i8>
      ! CHECK:     %[[V_390:[0-9]+]] = fir.load %[[V_389]] : !fir.ref<i8>
      ! CHECK:     %[[V_391:[0-9]+]] = fir.convert %[[V_390]] : (i8) -> i32
      ! CHECK:     %[[V_392:[0-9]+]] = fir.call @_FortranAMapException(%[[V_391]]) fastmath<contract> : (i32) -> i32
      ! CHECK:     fir.if %false{{[_0-9]*}} {
      ! CHECK:       %[[V_692:[0-9]+]] = fir.call @feraiseexcept(%[[V_392]]) fastmath<contract> : (i32) -> i32
      ! CHECK:     } else {
      ! CHECK:       %[[V_692:[0-9]+]] = fir.call @feclearexcept(%[[V_392]]) fastmath<contract> : (i32) -> i32
      ! CHECK:     }
      ! CHECK:     %[[V_393:[0-9]+]] = fir.load %[[V_17]] : !fir.ref<f32>
      ! CHECK:     %[[V_394:[0-9]+]] = fir.load %[[V_19]] : !fir.ref<f32>
      ! CHECK:     %[[V_395:[0-9]+]] = math.copysign %[[V_393]], %cst{{[_0-9]*}} fastmath<contract> : f32
      ! CHECK:     %[[V_396:[0-9]+]] = math.copysign %[[V_394]], %cst{{[_0-9]*}} fastmath<contract> : f32
      ! CHECK:     %[[V_397:[0-9]+]] = arith.cmpf olt, %[[V_395]], %[[V_396]] {{.*}} : f32
      ! CHECK:     %[[V_398:[0-9]+]] = fir.if %[[V_397]] -> (f32) {
      ! CHECK:       fir.result %[[V_394]] : f32
      ! CHECK:     } else {
      ! CHECK:       %[[V_692:[0-9]+]] = arith.cmpf ogt, %[[V_395]], %[[V_396]] {{.*}} : f32
      ! CHECK:       %[[V_693:[0-9]+]] = fir.if %[[V_692]] -> (f32) {
      ! CHECK:         fir.result %[[V_393]] : f32
      ! CHECK:       } else {
      ! CHECK:         %[[V_694:[0-9]+]] = arith.cmpf oeq, %[[V_395]], %[[V_396]] {{.*}} : f32
      ! CHECK:         %[[V_695:[0-9]+]] = fir.if %[[V_694]] -> (f32) {
      ! CHECK:           %[[V_696:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_393]]) <{bit = 960 : i32}> : (f32) -> i1
      ! CHECK:           %[[V_697:[0-9]+]] = arith.select %[[V_696]], %[[V_393]], %[[V_394]] : f32
      ! CHECK:           fir.result %[[V_697]] : f32
      ! CHECK:         } else {
      ! CHECK:           %[[V_696:[0-9]+]] = arith.cmpf ord, %[[V_393]], %[[V_393]] {{.*}} : f32
      ! CHECK:           %[[V_697:[0-9]+]] = arith.cmpf ord, %[[V_394]], %[[V_394]] {{.*}} : f32
      ! CHECK:           %[[V_698:[0-9]+]] = fir.coordinate_of %[[V_92]], %c2{{.*}} : (!fir.ref<!fir.array<12xi32>>, i8) -> !fir.ref<i32>
      ! CHECK:           %[[V_699:[0-9]+]] = fir.load %[[V_698]] : !fir.ref<i32>
      ! CHECK:           %[[V_700:[0-9]+]] = arith.bitcast %[[V_699]] : i32 to f32
      ! CHECK:           %[[V_701:[0-9]+]] = arith.select %[[V_697]], %[[V_394]], %[[V_700]] : f32
      ! CHECK:           %[[V_702:[0-9]+]] = arith.select %[[V_696]], %[[V_393]], %[[V_701]] : f32
      ! CHECK-DAG:       %[[V_703:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_394]]) <{bit = 1 : i32}> : (f32) -> i1
      ! CHECK-DAG:       %[[V_704:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_393]]) <{bit = 1 : i32}> : (f32) -> i1
      ! CHECK:           %[[V_705:[0-9]+]] = arith.ori %[[V_704]], %[[V_703]] : i1
      ! CHECK:           fir.if %[[V_705]] {
      ! CHECK:             %[[V_706:[0-9]+]] = fir.call @_FortranAMapException(%c1{{.*}}) fastmath<contract> : (i32) -> i32
      ! CHECK:             %[[V_707:[0-9]+]] = fir.call @feraiseexcept(%[[V_706]]) fastmath<contract> : (i32) -> i32
      ! CHECK:           }
      ! CHECK:           fir.result %[[V_702]] : f32
      ! CHECK:         }
      ! CHECK:         fir.result %[[V_695]] : f32
      ! CHECK:       }
      ! CHECK:       fir.result %[[V_693]] : f32
      ! CHECK:     }
      ! CHECK:     fir.store %[[V_398]] to %[[V_83]] : !fir.ref<f32>
      ! CHECK:     %[[V_399:[0-9]+]] = fir.declare %[[V_201]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro._QM__fortran_builtinsT__builtin_ieee_flag_type.10"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>
      ! CHECK:     %[[V_400:[0-9]+]] = fir.coordinate_of %[[V_399]], %[[V_203]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>, !fir.field) -> !fir.ref<i8>
      ! CHECK:     %[[V_401:[0-9]+]] = fir.load %[[V_400]] : !fir.ref<i8>
      ! CHECK:     %[[V_402:[0-9]+]] = fir.convert %[[V_401]] : (i8) -> i32
      ! CHECK:     %[[V_403:[0-9]+]] = fir.call @_FortranAMapException(%[[V_402]]) fastmath<contract> : (i32) -> i32
      ! CHECK:     %[[V_404:[0-9]+]] = fir.call @fetestexcept(%[[V_403]]) fastmath<contract> : (i32) -> i32
      ! CHECK:     %[[V_405:[0-9]+]] = arith.cmpi ne, %[[V_404]], %c0{{.*}} : i32
      ! CHECK:     %[[V_406:[0-9]+]] = fir.convert %[[V_405]] : (i1) -> !fir.logical<4>
      ! CHECK:     fir.store %[[V_406]] to %[[V_21]] : !fir.ref<!fir.logical<4>>
      call ieee_set_flag(ieee_invalid, .false.)
      r = ieee_max_num_mag(a, b)
      call ieee_get_flag(ieee_invalid, flag_value)
      write(*, 4) 'mag_num', a, a, b, b, r, flag_value, trim(tag(r))

      ! CHECK:     %[[V_449:[0-9]+]] = fir.declare %[[V_201]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro._QM__fortran_builtinsT__builtin_ieee_flag_type.10"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>
      ! CHECK:     %[[V_450:[0-9]+]] = fir.coordinate_of %[[V_449]], %[[V_203]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>, !fir.field) -> !fir.ref<i8>
      ! CHECK:     %[[V_451:[0-9]+]] = fir.load %[[V_450]] : !fir.ref<i8>
      ! CHECK:     %[[V_452:[0-9]+]] = fir.convert %[[V_451]] : (i8) -> i32
      ! CHECK:     %[[V_453:[0-9]+]] = fir.call @_FortranAMapException(%[[V_452]]) fastmath<contract> : (i32) -> i32
      ! CHECK:     fir.if %false{{[_0-9]*}} {
      ! CHECK:       %[[V_692:[0-9]+]] = fir.call @feraiseexcept(%[[V_453]]) fastmath<contract> : (i32) -> i32
      ! CHECK:     } else {
      ! CHECK:       %[[V_692:[0-9]+]] = fir.call @feclearexcept(%[[V_453]]) fastmath<contract> : (i32) -> i32
      ! CHECK:     }
      ! CHECK:     %[[V_454:[0-9]+]] = fir.load %[[V_17]] : !fir.ref<f32>
      ! CHECK:     %[[V_455:[0-9]+]] = fir.load %[[V_19]] : !fir.ref<f32>
      ! CHECK:     %[[V_456:[0-9]+]] = arith.cmpf olt, %[[V_454]], %[[V_455]] {{.*}} : f32
      ! CHECK:     %[[V_457:[0-9]+]] = fir.if %[[V_456]] -> (f32) {
      ! CHECK:       fir.result %[[V_454]] : f32
      ! CHECK:     } else {
      ! CHECK:       %[[V_692:[0-9]+]] = arith.cmpf ogt, %[[V_454]], %[[V_455]] {{.*}} : f32
      ! CHECK:       %[[V_693:[0-9]+]] = fir.if %[[V_692]] -> (f32) {
      ! CHECK:         fir.result %[[V_455]] : f32
      ! CHECK:       } else {
      ! CHECK:         %[[V_694:[0-9]+]] = arith.cmpf oeq, %[[V_454]], %[[V_455]] {{.*}} : f32
      ! CHECK:         %[[V_695:[0-9]+]] = fir.if %[[V_694]] -> (f32) {
      ! CHECK:           %[[V_696:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_454]]) <{bit = 60 : i32}> : (f32) -> i1
      ! CHECK:           %[[V_697:[0-9]+]] = arith.select %[[V_696]], %[[V_454]], %[[V_455]] : f32
      ! CHECK:           fir.result %[[V_697]] : f32
      ! CHECK:         } else {
      ! CHECK:           %[[V_696:[0-9]+]] = fir.coordinate_of %[[V_92]], %c2{{.*}} : (!fir.ref<!fir.array<12xi32>>, i8) -> !fir.ref<i32>
      ! CHECK:           %[[V_697:[0-9]+]] = fir.load %[[V_696]] : !fir.ref<i32>
      ! CHECK:           %[[V_698:[0-9]+]] = arith.bitcast %[[V_697]] : i32 to f32
      ! CHECK-DAG:       %[[V_699:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_455]]) <{bit = 1 : i32}> : (f32) -> i1
      ! CHECK-DAG:       %[[V_700:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_454]]) <{bit = 1 : i32}> : (f32) -> i1
      ! CHECK:           %[[V_701:[0-9]+]] = arith.ori %[[V_700]], %[[V_699]] : i1
      ! CHECK:           fir.if %[[V_701]] {
      ! CHECK:             %[[V_702:[0-9]+]] = fir.call @_FortranAMapException(%c1{{.*}}) fastmath<contract> : (i32) -> i32
      ! CHECK:             %[[V_703:[0-9]+]] = fir.call @feraiseexcept(%[[V_702]]) fastmath<contract> : (i32) -> i32
      ! CHECK:           }
      ! CHECK:           fir.result %[[V_698]] : f32
      ! CHECK:         }
      ! CHECK:         fir.result %[[V_695]] : f32
      ! CHECK:       }
      ! CHECK:       fir.result %[[V_693]] : f32
      ! CHECK:     }
      ! CHECK:     fir.store %[[V_457]] to %[[V_83]] : !fir.ref<f32>
      ! CHECK:     %[[V_458:[0-9]+]] = fir.declare %[[V_201]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro._QM__fortran_builtinsT__builtin_ieee_flag_type.10"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>
      ! CHECK:     %[[V_459:[0-9]+]] = fir.coordinate_of %[[V_458]], %[[V_203]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>, !fir.field) -> !fir.ref<i8>
      ! CHECK:     %[[V_460:[0-9]+]] = fir.load %[[V_459]] : !fir.ref<i8>
      ! CHECK:     %[[V_461:[0-9]+]] = fir.convert %[[V_460]] : (i8) -> i32
      ! CHECK:     %[[V_462:[0-9]+]] = fir.call @_FortranAMapException(%[[V_461]]) fastmath<contract> : (i32) -> i32
      ! CHECK:     %[[V_463:[0-9]+]] = fir.call @fetestexcept(%[[V_462]]) fastmath<contract> : (i32) -> i32
      ! CHECK:     %[[V_464:[0-9]+]] = arith.cmpi ne, %[[V_463]], %c0{{.*}} : i32
      ! CHECK:     %[[V_465:[0-9]+]] = fir.convert %[[V_464]] : (i1) -> !fir.logical<4>
      ! CHECK:     fir.store %[[V_465]] to %[[V_21]] : !fir.ref<!fir.logical<4>>
      call ieee_set_flag(ieee_invalid, .false.)
      r = ieee_min(a, b)
      call ieee_get_flag(ieee_invalid, flag_value)
      write(*, 4) 'min    ', a, a, b, b, r, flag_value, trim(tag(r))

      ! CHECK:     %[[V_508:[0-9]+]] = fir.declare %[[V_201]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro._QM__fortran_builtinsT__builtin_ieee_flag_type.10"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>
      ! CHECK:     %[[V_509:[0-9]+]] = fir.coordinate_of %[[V_508]], %[[V_203]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>, !fir.field) -> !fir.ref<i8>
      ! CHECK:     %[[V_510:[0-9]+]] = fir.load %[[V_509]] : !fir.ref<i8>
      ! CHECK:     %[[V_511:[0-9]+]] = fir.convert %[[V_510]] : (i8) -> i32
      ! CHECK:     %[[V_512:[0-9]+]] = fir.call @_FortranAMapException(%[[V_511]]) fastmath<contract> : (i32) -> i32
      ! CHECK:     fir.if %false{{[_0-9]*}} {
      ! CHECK:       %[[V_692:[0-9]+]] = fir.call @feraiseexcept(%[[V_512]]) fastmath<contract> : (i32) -> i32
      ! CHECK:     } else {
      ! CHECK:       %[[V_692:[0-9]+]] = fir.call @feclearexcept(%[[V_512]]) fastmath<contract> : (i32) -> i32
      ! CHECK:     }
      ! CHECK:     %[[V_513:[0-9]+]] = fir.load %[[V_17]] : !fir.ref<f32>
      ! CHECK:     %[[V_514:[0-9]+]] = fir.load %[[V_19]] : !fir.ref<f32>
      ! CHECK:     %[[V_515:[0-9]+]] = math.copysign %[[V_513]], %cst{{[_0-9]*}} fastmath<contract> : f32
      ! CHECK:     %[[V_516:[0-9]+]] = math.copysign %[[V_514]], %cst{{[_0-9]*}} fastmath<contract> : f32
      ! CHECK:     %[[V_517:[0-9]+]] = arith.cmpf olt, %[[V_515]], %[[V_516]] {{.*}} : f32
      ! CHECK:     %[[V_518:[0-9]+]] = fir.if %[[V_517]] -> (f32) {
      ! CHECK:       fir.result %[[V_513]] : f32
      ! CHECK:     } else {
      ! CHECK:       %[[V_692:[0-9]+]] = arith.cmpf ogt, %[[V_515]], %[[V_516]] {{.*}} : f32
      ! CHECK:       %[[V_693:[0-9]+]] = fir.if %[[V_692]] -> (f32) {
      ! CHECK:         fir.result %[[V_514]] : f32
      ! CHECK:       } else {
      ! CHECK:         %[[V_694:[0-9]+]] = arith.cmpf oeq, %[[V_515]], %[[V_516]] {{.*}} : f32
      ! CHECK:         %[[V_695:[0-9]+]] = fir.if %[[V_694]] -> (f32) {
      ! CHECK:           %[[V_696:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_513]]) <{bit = 60 : i32}> : (f32) -> i1
      ! CHECK:           %[[V_697:[0-9]+]] = arith.select %[[V_696]], %[[V_513]], %[[V_514]] : f32
      ! CHECK:           fir.result %[[V_697]] : f32
      ! CHECK:         } else {
      ! CHECK:           %[[V_696:[0-9]+]] = fir.coordinate_of %[[V_92]], %c2{{.*}} : (!fir.ref<!fir.array<12xi32>>, i8) -> !fir.ref<i32>
      ! CHECK:           %[[V_697:[0-9]+]] = fir.load %[[V_696]] : !fir.ref<i32>
      ! CHECK:           %[[V_698:[0-9]+]] = arith.bitcast %[[V_697]] : i32 to f32
      ! CHECK-DAG:       %[[V_699:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_514]]) <{bit = 1 : i32}> : (f32) -> i1
      ! CHECK-DAG:       %[[V_700:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_513]]) <{bit = 1 : i32}> : (f32) -> i1
      ! CHECK:           %[[V_701:[0-9]+]] = arith.ori %[[V_700]], %[[V_699]] : i1
      ! CHECK:           fir.if %[[V_701]] {
      ! CHECK:             %[[V_702:[0-9]+]] = fir.call @_FortranAMapException(%c1{{.*}}) fastmath<contract> : (i32) -> i32
      ! CHECK:             %[[V_703:[0-9]+]] = fir.call @feraiseexcept(%[[V_702]]) fastmath<contract> : (i32) -> i32
      ! CHECK:           }
      ! CHECK:           fir.result %[[V_698]] : f32
      ! CHECK:         }
      ! CHECK:         fir.result %[[V_695]] : f32
      ! CHECK:       }
      ! CHECK:       fir.result %[[V_693]] : f32
      ! CHECK:     }
      ! CHECK:     fir.store %[[V_518]] to %[[V_83]] : !fir.ref<f32>
      ! CHECK:     %[[V_519:[0-9]+]] = fir.declare %[[V_201]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro._QM__fortran_builtinsT__builtin_ieee_flag_type.10"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>
      ! CHECK:     %[[V_520:[0-9]+]] = fir.coordinate_of %[[V_519]], %[[V_203]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>, !fir.field) -> !fir.ref<i8>
      ! CHECK:     %[[V_521:[0-9]+]] = fir.load %[[V_520]] : !fir.ref<i8>
      ! CHECK:     %[[V_522:[0-9]+]] = fir.convert %[[V_521]] : (i8) -> i32
      ! CHECK:     %[[V_523:[0-9]+]] = fir.call @_FortranAMapException(%[[V_522]]) fastmath<contract> : (i32) -> i32
      ! CHECK:     %[[V_524:[0-9]+]] = fir.call @fetestexcept(%[[V_523]]) fastmath<contract> : (i32) -> i32
      ! CHECK:     %[[V_525:[0-9]+]] = arith.cmpi ne, %[[V_524]], %c0{{.*}} : i32
      ! CHECK:     %[[V_526:[0-9]+]] = fir.convert %[[V_525]] : (i1) -> !fir.logical<4>
      ! CHECK:     fir.store %[[V_526]] to %[[V_21]] : !fir.ref<!fir.logical<4>>
      call ieee_set_flag(ieee_invalid, .false.)
      r = ieee_min_mag(a, b)
      call ieee_get_flag(ieee_invalid, flag_value)
      write(*, 4) 'mig    ', a, a, b, b, r, flag_value, trim(tag(r))

      ! CHECK:     %[[V_569:[0-9]+]] = fir.declare %[[V_201]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro._QM__fortran_builtinsT__builtin_ieee_flag_type.10"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>
      ! CHECK:     %[[V_570:[0-9]+]] = fir.coordinate_of %[[V_569]], %[[V_203]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>, !fir.field) -> !fir.ref<i8>
      ! CHECK:     %[[V_571:[0-9]+]] = fir.load %[[V_570]] : !fir.ref<i8>
      ! CHECK:     %[[V_572:[0-9]+]] = fir.convert %[[V_571]] : (i8) -> i32
      ! CHECK:     %[[V_573:[0-9]+]] = fir.call @_FortranAMapException(%[[V_572]]) fastmath<contract> : (i32) -> i32
      ! CHECK:     fir.if %false{{[_0-9]*}} {
      ! CHECK:       %[[V_692:[0-9]+]] = fir.call @feraiseexcept(%[[V_573]]) fastmath<contract> : (i32) -> i32
      ! CHECK:     } else {
      ! CHECK:       %[[V_692:[0-9]+]] = fir.call @feclearexcept(%[[V_573]]) fastmath<contract> : (i32) -> i32
      ! CHECK:     }
      ! CHECK:     %[[V_574:[0-9]+]] = fir.load %[[V_17]] : !fir.ref<f32>
      ! CHECK:     %[[V_575:[0-9]+]] = fir.load %[[V_19]] : !fir.ref<f32>
      ! CHECK:     %[[V_576:[0-9]+]] = arith.cmpf olt, %[[V_574]], %[[V_575]] {{.*}} : f32
      ! CHECK:     %[[V_577:[0-9]+]] = fir.if %[[V_576]] -> (f32) {
      ! CHECK:       fir.result %[[V_574]] : f32
      ! CHECK:     } else {
      ! CHECK:       %[[V_692:[0-9]+]] = arith.cmpf ogt, %[[V_574]], %[[V_575]] {{.*}} : f32
      ! CHECK:       %[[V_693:[0-9]+]] = fir.if %[[V_692]] -> (f32) {
      ! CHECK:         fir.result %[[V_575]] : f32
      ! CHECK:       } else {
      ! CHECK:         %[[V_694:[0-9]+]] = arith.cmpf oeq, %[[V_574]], %[[V_575]] {{.*}} : f32
      ! CHECK:         %[[V_695:[0-9]+]] = fir.if %[[V_694]] -> (f32) {
      ! CHECK:           %[[V_696:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_574]]) <{bit = 60 : i32}> : (f32) -> i1
      ! CHECK:           %[[V_697:[0-9]+]] = arith.select %[[V_696]], %[[V_574]], %[[V_575]] : f32
      ! CHECK:           fir.result %[[V_697]] : f32
      ! CHECK:         } else {
      ! CHECK:           %[[V_696:[0-9]+]] = arith.cmpf ord, %[[V_574]], %[[V_574]] {{.*}} : f32
      ! CHECK:           %[[V_697:[0-9]+]] = arith.cmpf ord, %[[V_575]], %[[V_575]] {{.*}} : f32
      ! CHECK:           %[[V_698:[0-9]+]] = fir.coordinate_of %[[V_92]], %c2{{.*}} : (!fir.ref<!fir.array<12xi32>>, i8) -> !fir.ref<i32>
      ! CHECK:           %[[V_699:[0-9]+]] = fir.load %[[V_698]] : !fir.ref<i32>
      ! CHECK:           %[[V_700:[0-9]+]] = arith.bitcast %[[V_699]] : i32 to f32
      ! CHECK:           %[[V_701:[0-9]+]] = arith.select %[[V_697]], %[[V_575]], %[[V_700]] : f32
      ! CHECK:           %[[V_702:[0-9]+]] = arith.select %[[V_696]], %[[V_574]], %[[V_701]] : f32
      ! CHECK-DAG:       %[[V_703:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_575]]) <{bit = 1 : i32}> : (f32) -> i1
      ! CHECK-DAG:       %[[V_704:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_574]]) <{bit = 1 : i32}> : (f32) -> i1
      ! CHECK:           %[[V_705:[0-9]+]] = arith.ori %[[V_704]], %[[V_703]] : i1
      ! CHECK:           fir.if %[[V_705]] {
      ! CHECK:             %[[V_706:[0-9]+]] = fir.call @_FortranAMapException(%c1{{.*}}) fastmath<contract> : (i32) -> i32
      ! CHECK:             %[[V_707:[0-9]+]] = fir.call @feraiseexcept(%[[V_706]]) fastmath<contract> : (i32) -> i32
      ! CHECK:           }
      ! CHECK:           fir.result %[[V_702]] : f32
      ! CHECK:         }
      ! CHECK:         fir.result %[[V_695]] : f32
      ! CHECK:       }
      ! CHECK:       fir.result %[[V_693]] : f32
      ! CHECK:     }
      ! CHECK:     fir.store %[[V_577]] to %[[V_83]] : !fir.ref<f32>
      ! CHECK:     %[[V_578:[0-9]+]] = fir.declare %[[V_201]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro._QM__fortran_builtinsT__builtin_ieee_flag_type.10"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>
      ! CHECK:     %[[V_579:[0-9]+]] = fir.coordinate_of %[[V_578]], %[[V_203]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>, !fir.field) -> !fir.ref<i8>
      ! CHECK:     %[[V_580:[0-9]+]] = fir.load %[[V_579]] : !fir.ref<i8>
      ! CHECK:     %[[V_581:[0-9]+]] = fir.convert %[[V_580]] : (i8) -> i32
      ! CHECK:     %[[V_582:[0-9]+]] = fir.call @_FortranAMapException(%[[V_581]]) fastmath<contract> : (i32) -> i32
      ! CHECK:     %[[V_583:[0-9]+]] = fir.call @fetestexcept(%[[V_582]]) fastmath<contract> : (i32) -> i32
      ! CHECK:     %[[V_584:[0-9]+]] = arith.cmpi ne, %[[V_583]], %c0{{.*}} : i32
      ! CHECK:     %[[V_585:[0-9]+]] = fir.convert %[[V_584]] : (i1) -> !fir.logical<4>
      ! CHECK:     fir.store %[[V_585]] to %[[V_21]] : !fir.ref<!fir.logical<4>>
      call ieee_set_flag(ieee_invalid, .false.)
      r = ieee_min_num(a, b)
      call ieee_get_flag(ieee_invalid, flag_value)
      write(*, 4) 'min_num', a, a, b, b, r, flag_value, trim(tag(r))

      ! CHECK:     %[[V_628:[0-9]+]] = fir.declare %[[V_201]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro._QM__fortran_builtinsT__builtin_ieee_flag_type.10"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>
      ! CHECK:     %[[V_629:[0-9]+]] = fir.coordinate_of %[[V_628]], %[[V_203]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>, !fir.field) -> !fir.ref<i8>
      ! CHECK:     %[[V_630:[0-9]+]] = fir.load %[[V_629]] : !fir.ref<i8>
      ! CHECK:     %[[V_631:[0-9]+]] = fir.convert %[[V_630]] : (i8) -> i32
      ! CHECK:     %[[V_632:[0-9]+]] = fir.call @_FortranAMapException(%[[V_631]]) fastmath<contract> : (i32) -> i32
      ! CHECK:     fir.if %false{{[_0-9]*}} {
      ! CHECK:       %[[V_692:[0-9]+]] = fir.call @feraiseexcept(%[[V_632]]) fastmath<contract> : (i32) -> i32
      ! CHECK:     } else {
      ! CHECK:       %[[V_692:[0-9]+]] = fir.call @feclearexcept(%[[V_632]]) fastmath<contract> : (i32) -> i32
      ! CHECK:     }
      ! CHECK:     %[[V_633:[0-9]+]] = fir.load %[[V_17]] : !fir.ref<f32>
      ! CHECK:     %[[V_634:[0-9]+]] = fir.load %[[V_19]] : !fir.ref<f32>
      ! CHECK:     %[[V_635:[0-9]+]] = math.copysign %[[V_633]], %cst{{[_0-9]*}} fastmath<contract> : f32
      ! CHECK:     %[[V_636:[0-9]+]] = math.copysign %[[V_634]], %cst{{[_0-9]*}} fastmath<contract> : f32
      ! CHECK:     %[[V_637:[0-9]+]] = arith.cmpf olt, %[[V_635]], %[[V_636]] {{.*}} : f32
      ! CHECK:     %[[V_638:[0-9]+]] = fir.if %[[V_637]] -> (f32) {
      ! CHECK:       fir.result %[[V_633]] : f32
      ! CHECK:     } else {
      ! CHECK:       %[[V_692:[0-9]+]] = arith.cmpf ogt, %[[V_635]], %[[V_636]] {{.*}} : f32
      ! CHECK:       %[[V_693:[0-9]+]] = fir.if %[[V_692]] -> (f32) {
      ! CHECK:         fir.result %[[V_634]] : f32
      ! CHECK:       } else {
      ! CHECK:         %[[V_694:[0-9]+]] = arith.cmpf oeq, %[[V_635]], %[[V_636]] {{.*}} : f32
      ! CHECK:         %[[V_695:[0-9]+]] = fir.if %[[V_694]] -> (f32) {
      ! CHECK:           %[[V_696:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_633]]) <{bit = 60 : i32}> : (f32) -> i1
      ! CHECK:           %[[V_697:[0-9]+]] = arith.select %[[V_696]], %[[V_633]], %[[V_634]] : f32
      ! CHECK:           fir.result %[[V_697]] : f32
      ! CHECK:         } else {
      ! CHECK:           %[[V_696:[0-9]+]] = arith.cmpf ord, %[[V_633]], %[[V_633]] {{.*}} : f32
      ! CHECK:           %[[V_697:[0-9]+]] = arith.cmpf ord, %[[V_634]], %[[V_634]] {{.*}} : f32
      ! CHECK:           %[[V_698:[0-9]+]] = fir.coordinate_of %[[V_92]], %c2{{.*}} : (!fir.ref<!fir.array<12xi32>>, i8) -> !fir.ref<i32>
      ! CHECK:           %[[V_699:[0-9]+]] = fir.load %[[V_698]] : !fir.ref<i32>
      ! CHECK:           %[[V_700:[0-9]+]] = arith.bitcast %[[V_699]] : i32 to f32
      ! CHECK:           %[[V_701:[0-9]+]] = arith.select %[[V_697]], %[[V_634]], %[[V_700]] : f32
      ! CHECK:           %[[V_702:[0-9]+]] = arith.select %[[V_696]], %[[V_633]], %[[V_701]] : f32
      ! CHECK-DAG:       %[[V_703:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_634]]) <{bit = 1 : i32}> : (f32) -> i1
      ! CHECK-DAG:       %[[V_704:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_633]]) <{bit = 1 : i32}> : (f32) -> i1
      ! CHECK:           %[[V_705:[0-9]+]] = arith.ori %[[V_704]], %[[V_703]] : i1
      ! CHECK:           fir.if %[[V_705]] {
      ! CHECK:             %[[V_706:[0-9]+]] = fir.call @_FortranAMapException(%c1{{.*}}) fastmath<contract> : (i32) -> i32
      ! CHECK:             %[[V_707:[0-9]+]] = fir.call @feraiseexcept(%[[V_706]]) fastmath<contract> : (i32) -> i32
      ! CHECK:           }
      ! CHECK:           fir.result %[[V_702]] : f32
      ! CHECK:         }
      ! CHECK:         fir.result %[[V_695]] : f32
      ! CHECK:       }
      ! CHECK:       fir.result %[[V_693]] : f32
      ! CHECK:     }
      ! CHECK:     fir.store %[[V_638]] to %[[V_83]] : !fir.ref<f32>
      ! CHECK:     %[[V_639:[0-9]+]] = fir.declare %[[V_201]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro._QM__fortran_builtinsT__builtin_ieee_flag_type.10"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>
      ! CHECK:     %[[V_640:[0-9]+]] = fir.coordinate_of %[[V_639]], %[[V_203]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>, !fir.field) -> !fir.ref<i8>
      ! CHECK:     %[[V_641:[0-9]+]] = fir.load %[[V_640]] : !fir.ref<i8>
      ! CHECK:     %[[V_642:[0-9]+]] = fir.convert %[[V_641]] : (i8) -> i32
      ! CHECK:     %[[V_643:[0-9]+]] = fir.call @_FortranAMapException(%[[V_642]]) fastmath<contract> : (i32) -> i32
      ! CHECK:     %[[V_644:[0-9]+]] = fir.call @fetestexcept(%[[V_643]]) fastmath<contract> : (i32) -> i32
      ! CHECK:     %[[V_645:[0-9]+]] = arith.cmpi ne, %[[V_644]], %c0{{.*}} : i32
      ! CHECK:     %[[V_646:[0-9]+]] = fir.convert %[[V_645]] : (i1) -> !fir.logical<4>
      ! CHECK:     fir.store %[[V_646]] to %[[V_21]] : !fir.ref<!fir.logical<4>>
      call ieee_set_flag(ieee_invalid, .false.)
      r = ieee_min_num_mag(a, b)
      call ieee_get_flag(ieee_invalid, flag_value)
      write(*, 4) 'mig_num', a, a, b, b, r, flag_value, trim(tag(r))
    enddo
  enddo
end
