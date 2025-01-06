! RUN: bbc -emit-fir -o - %s | FileCheck %s

! CHECK-LABEL: c.func @_QQmain

  ! CHECK:     %[[V_0:[0-9]+]] = fir.address_of(@_QM__fortran_ieee_exceptionsECieee_all) : !fir.ref<!fir.array<5x!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>>
  ! CHECK:     %[[V_1:[0-9]+]] = fir.shape %c5{{.*}} : (index) -> !fir.shape<1>
  ! CHECK:     %[[V_2:[0-9]+]] = fir.declare %[[V_0]](%[[V_1]]) {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QM__fortran_ieee_exceptionsECieee_all"} : (!fir.ref<!fir.array<5x!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>>, !fir.shape<1>) -> !fir.ref<!fir.array<5x!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>>
  ! CHECK:     %[[V_53:[0-9]+]] = fir.address_of(@_QM__fortran_ieee_exceptionsECieee_usual) : !fir.ref<!fir.array<3x!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>>
  ! CHECK:     %[[V_54:[0-9]+]] = fir.shape %c3{{.*}} : (index) -> !fir.shape<1>
  ! CHECK:     %[[V_55:[0-9]+]] = fir.declare %[[V_53]](%[[V_54]]) {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QM__fortran_ieee_exceptionsECieee_usual"} : (!fir.ref<!fir.array<3x!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>>, !fir.shape<1>) -> !fir.ref<!fir.array<3x!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>>
  use ieee_arithmetic

  ! CHECK:     %[[V_56:[0-9]+]] = fir.alloca !fir.logical<4> {bindc_name = "v", uniq_name = "_QFEv"}
  ! CHECK:     %[[V_57:[0-9]+]] = fir.declare %[[V_56]] {uniq_name = "_QFEv"} : (!fir.ref<!fir.logical<4>>) -> !fir.ref<!fir.logical<4>>
  ! CHECK:     %[[V_58:[0-9]+]] = fir.alloca !fir.array<2x!fir.logical<4>> {bindc_name = "v2", uniq_name = "_QFEv2"}
  ! CHECK:     %[[V_59:[0-9]+]] = fir.shape %c2{{.*}} : (index) -> !fir.shape<1>
  ! CHECK:     %[[V_60:[0-9]+]] = fir.declare %[[V_58]](%[[V_59]]) {uniq_name = "_QFEv2"} : (!fir.ref<!fir.array<2x!fir.logical<4>>>, !fir.shape<1>) -> !fir.ref<!fir.array<2x!fir.logical<4>>>
  ! CHECK:     %[[V_61:[0-9]+]] = fir.alloca !fir.array<5x!fir.logical<4>> {bindc_name = "v_all", uniq_name = "_QFEv_all"}
  ! CHECK:     %[[V_62:[0-9]+]] = fir.declare %[[V_61]](%[[V_1]]) {uniq_name = "_QFEv_all"} : (!fir.ref<!fir.array<5x!fir.logical<4>>>, !fir.shape<1>) -> !fir.ref<!fir.array<5x!fir.logical<4>>>
  ! CHECK:     %[[V_63:[0-9]+]] = fir.alloca !fir.array<3x!fir.logical<4>> {bindc_name = "v_usual", uniq_name = "_QFEv_usual"}
  ! CHECK:     %[[V_64:[0-9]+]] = fir.declare %[[V_63]](%[[V_54]]) {uniq_name = "_QFEv_usual"} : (!fir.ref<!fir.array<3x!fir.logical<4>>>, !fir.shape<1>) -> !fir.ref<!fir.array<3x!fir.logical<4>>>
  logical :: v, v2(2), v_usual(size(ieee_usual)), v_all(size(ieee_all))
  real :: x(100,20,10)

  ! CHECK:     %[[V_67:[0-9]+]] = fir.call @_FortranAioBeginExternalListOutput
  print*, 'Flag'

  ! CHECK:     %[[V_74:[0-9]+]] = fir.call @_FortranAioBeginExternalListOutput
  ! CHECK:     %[[V_87:[0-9]+]] = fir.call @_FortranAioOutputLogical(%[[V_74]], %true) fastmath<contract> : (!fir.ref<i8>, i1) -> i1
  ! CHECK:     %[[V_93:[0-9]+]] = fir.call @_FortranAioOutputLogical(%[[V_74]], %true) fastmath<contract> : (!fir.ref<i8>, i1) -> i1
  print*, 'support invalid: ', &
    ieee_support_flag(ieee_invalid), ieee_support_flag(ieee_invalid, x)

  ! CHECK:     %[[V_80:[0-9]+]] = fir.address_of(@_QQro._QM__fortran_builtinsT__builtin_ieee_flag_type.0) : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>
  ! CHECK:     %[[V_95:[0-9]+]] = fir.declare %[[V_80]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro._QM__fortran_builtinsT__builtin_ieee_flag_type.0"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>
  ! CHECK:     %[[V_82:[0-9]+]] = fir.field_index _QM__fortran_builtinsT__builtin_ieee_flag_type.flag, !fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>
  ! CHECK:     %[[V_96:[0-9]+]] = fir.coordinate_of %[[V_95]], %[[V_82]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>, !fir.field) -> !fir.ref<i8>
  ! CHECK:     %[[V_97:[0-9]+]] = fir.load %[[V_96]] : !fir.ref<i8>
  ! CHECK:     %[[V_98:[0-9]+]] = fir.convert %[[V_97]] : (i8) -> i32
  ! CHECK:     %[[V_99:[0-9]+]] = fir.call @_FortranAMapException(%[[V_98]]) fastmath<contract> : (i32) -> i32
  ! CHECK:     fir.if %false{{[_0-9]*}} {
  ! CHECK:       %[[V_310:[0-9]+]] = fir.call @feraiseexcept(%[[V_99]]) fastmath<contract> : (i32) -> i32
  ! CHECK:     } else {
  ! CHECK:       %[[V_310:[0-9]+]] = fir.call @feclearexcept(%[[V_99]]) fastmath<contract> : (i32) -> i32
  ! CHECK:     }
  call ieee_set_flag(ieee_invalid, .false.)

  ! CHECK:     %[[V_100:[0-9]+]] = fir.declare %[[V_80]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro._QM__fortran_builtinsT__builtin_ieee_flag_type.0"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>
  ! CHECK:     %[[V_101:[0-9]+]] = fir.coordinate_of %[[V_100]], %[[V_82]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>, !fir.field) -> !fir.ref<i8>
  ! CHECK:     %[[V_102:[0-9]+]] = fir.load %[[V_101]] : !fir.ref<i8>
  ! CHECK:     %[[V_103:[0-9]+]] = fir.convert %[[V_102]] : (i8) -> i32
  ! CHECK:     %[[V_104:[0-9]+]] = fir.call @_FortranAMapException(%[[V_103]]) fastmath<contract> : (i32) -> i32
  ! CHECK:     %[[V_105:[0-9]+]] = fir.call @fetestexcept(%[[V_104]]) fastmath<contract> : (i32) -> i32
  ! CHECK:     %[[V_106:[0-9]+]] = arith.cmpi ne, %[[V_105]], %c0{{.*}} : i32
  ! CHECK:     %[[V_107:[0-9]+]] = fir.convert %[[V_106]] : (i1) -> !fir.logical<4>
  ! CHECK:     fir.store %[[V_107]] to %[[V_57]] : !fir.ref<!fir.logical<4>>
  call ieee_get_flag(ieee_invalid, v)

  ! CHECK:     %[[V_108:[0-9]+]] = fir.call @_FortranAioBeginExternalListOutput
  print*, 'invalid[F]: ', v

  ! CHECK:     %[[V_118:[0-9]+]] = fir.declare %[[V_80]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro._QM__fortran_builtinsT__builtin_ieee_flag_type.0"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>
  ! CHECK:     %[[V_119:[0-9]+]] = fir.coordinate_of %[[V_118]], %[[V_82]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>, !fir.field) -> !fir.ref<i8>
  ! CHECK:     %[[V_120:[0-9]+]] = fir.load %[[V_119]] : !fir.ref<i8>
  ! CHECK:     %[[V_121:[0-9]+]] = fir.convert %[[V_120]] : (i8) -> i32
  ! CHECK:     %[[V_122:[0-9]+]] = fir.call @_FortranAMapException(%[[V_121]]) fastmath<contract> : (i32) -> i32
  ! CHECK:     fir.if %true{{[_0-9]*}} {
  ! CHECK:       %[[V_310:[0-9]+]] = fir.call @feraiseexcept(%[[V_122]]) fastmath<contract> : (i32) -> i32
  ! CHECK:     } else {
  ! CHECK:       %[[V_310:[0-9]+]] = fir.call @feclearexcept(%[[V_122]]) fastmath<contract> : (i32) -> i32
  ! CHECK:     }
  call ieee_set_flag(ieee_invalid, .true.)

  ! CHECK:     %[[V_123:[0-9]+]] = fir.declare %[[V_80]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro._QM__fortran_builtinsT__builtin_ieee_flag_type.0"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>
  ! CHECK:     %[[V_124:[0-9]+]] = fir.coordinate_of %[[V_123]], %[[V_82]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>, !fir.field) -> !fir.ref<i8>
  ! CHECK:     %[[V_125:[0-9]+]] = fir.load %[[V_124]] : !fir.ref<i8>
  ! CHECK:     %[[V_126:[0-9]+]] = fir.convert %[[V_125]] : (i8) -> i32
  ! CHECK:     %[[V_127:[0-9]+]] = fir.call @_FortranAMapException(%[[V_126]]) fastmath<contract> : (i32) -> i32
  ! CHECK:     %[[V_128:[0-9]+]] = fir.call @fetestexcept(%[[V_127]]) fastmath<contract> : (i32) -> i32
  ! CHECK:     %[[V_129:[0-9]+]] = arith.cmpi ne, %[[V_128]], %c0{{.*}} : i32
  ! CHECK:     %[[V_130:[0-9]+]] = fir.convert %[[V_129]] : (i1) -> !fir.logical<4>
  ! CHECK:     fir.store %[[V_130]] to %[[V_57]] : !fir.ref<!fir.logical<4>>
  call ieee_get_flag(ieee_invalid, v)

  ! CHECK:     %[[V_131:[0-9]+]] = fir.call @_FortranAioBeginExternalListOutput
  print*, 'invalid[T]: ', v

  ! CHECK:     %[[V_140:[0-9]+]] = fir.address_of(@_QQro.2x_QM__fortran_builtinsT__builtin_ieee_flag_type.1) : !fir.ref<!fir.array<2x!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>>
  ! CHECK:     %[[V_141:[0-9]+]] = fir.declare %[[V_140]](%[[V_59]]) {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro.2x_QM__fortran_builtinsT__builtin_ieee_flag_type.1"} : (!fir.ref<!fir.array<2x!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>>, !fir.shape<1>) -> !fir.ref<!fir.array<2x!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>>
  ! CHECK:     fir.do_loop %arg0 = %c1{{.*}} to %c2{{.*}} step %c1{{.*}} {
  ! CHECK:       %[[V_310:[0-9]+]] = fir.array_coor %[[V_141]](%[[V_59]]) %arg0 : (!fir.ref<!fir.array<2x!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>>, !fir.shape<1>, index) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>
  ! CHECK:       %[[V_311:[0-9]+]] = fir.coordinate_of %[[V_310]], %[[V_82]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>, !fir.field) -> !fir.ref<i8>
  ! CHECK:       %[[V_312:[0-9]+]] = fir.load %[[V_311]] : !fir.ref<i8>
  ! CHECK:       %[[V_313:[0-9]+]] = fir.convert %[[V_312]] : (i8) -> i32
  ! CHECK:       %[[V_314:[0-9]+]] = fir.call @_FortranAMapException(%[[V_313]]) fastmath<contract> : (i32) -> i32
  ! CHECK:       fir.if %false{{[_0-9]*}} {
  ! CHECK:         %[[V_315:[0-9]+]] = fir.call @feraiseexcept(%[[V_314]]) fastmath<contract> : (i32) -> i32
  ! CHECK:       } else {
  ! CHECK:         %[[V_315:[0-9]+]] = fir.call @feclearexcept(%[[V_314]]) fastmath<contract> : (i32) -> i32
  ! CHECK:       }
  ! CHECK:     }
  call ieee_set_flag([ieee_invalid, ieee_overflow], .false.)

  ! CHECK:     %[[V_142:[0-9]+]] = fir.address_of(@_QQro.2x_QM__fortran_builtinsT__builtin_ieee_flag_type.2) : !fir.ref<!fir.array<2x!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>>
  ! CHECK:     %[[V_143:[0-9]+]] = fir.declare %[[V_142]](%[[V_59]]) {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro.2x_QM__fortran_builtinsT__builtin_ieee_flag_type.2"} : (!fir.ref<!fir.array<2x!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>>, !fir.shape<1>) -> !fir.ref<!fir.array<2x!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>>
  ! CHECK:     fir.do_loop %arg0 = %c1{{.*}} to %c2{{.*}} step %c1{{.*}} {
  ! CHECK:       %[[V_310:[0-9]+]] = fir.array_coor %[[V_143]](%[[V_59]]) %arg0 : (!fir.ref<!fir.array<2x!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>>, !fir.shape<1>, index) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>
  ! CHECK:       %[[V_311:[0-9]+]] = fir.array_coor %[[V_60]](%[[V_59]]) %arg0 : (!fir.ref<!fir.array<2x!fir.logical<4>>>, !fir.shape<1>, index) -> !fir.ref<!fir.logical<4>>
  ! CHECK:       %[[V_312:[0-9]+]] = fir.coordinate_of %[[V_310]], %[[V_82]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>, !fir.field) -> !fir.ref<i8>
  ! CHECK:       %[[V_313:[0-9]+]] = fir.load %[[V_312]] : !fir.ref<i8>
  ! CHECK:       %[[V_314:[0-9]+]] = fir.convert %[[V_313]] : (i8) -> i32
  ! CHECK:       %[[V_315:[0-9]+]] = fir.call @_FortranAMapException(%[[V_314]]) fastmath<contract> : (i32) -> i32
  ! CHECK:       %[[V_316:[0-9]+]] = fir.call @fetestexcept(%[[V_315]]) fastmath<contract> : (i32) -> i32
  ! CHECK:       %[[V_317:[0-9]+]] = arith.cmpi ne, %[[V_316]], %c0{{.*}} : i32
  ! CHECK:       %[[V_318:[0-9]+]] = fir.convert %[[V_317]] : (i1) -> !fir.logical<4>
  ! CHECK:       fir.store %[[V_318]] to %[[V_311]] : !fir.ref<!fir.logical<4>>
  ! CHECK:     }
  call ieee_get_flag([ieee_overflow, ieee_invalid], v2)

  ! CHECK:     %[[V_144:[0-9]+]] = fir.call @_FortranAioBeginExternalListOutput
  print*, '[overflow[F], invalid[F]]: ', v2

  ! CHECK:     %[[V_154:[0-9]+]] = fir.declare %[[V_140]](%[[V_59]]) {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro.2x_QM__fortran_builtinsT__builtin_ieee_flag_type.1"} : (!fir.ref<!fir.array<2x!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>>, !fir.shape<1>) -> !fir.ref<!fir.array<2x!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>>
  ! CHECK:     %[[V_155:[0-9]+]] = fir.address_of(@_QQro.2xl4.3) : !fir.ref<!fir.array<2x!fir.logical<4>>>
  ! CHECK:     %[[V_156:[0-9]+]] = fir.declare %[[V_155]](%[[V_59]]) {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro.2xl4.3"} : (!fir.ref<!fir.array<2x!fir.logical<4>>>, !fir.shape<1>) -> !fir.ref<!fir.array<2x!fir.logical<4>>>
  ! CHECK:     fir.do_loop %arg0 = %c1{{.*}} to %c2{{.*}} step %c1{{.*}} {
  ! CHECK:       %[[V_310:[0-9]+]] = fir.array_coor %[[V_154]](%[[V_59]]) %arg0 : (!fir.ref<!fir.array<2x!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>>, !fir.shape<1>, index) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>
  ! CHECK:       %[[V_311:[0-9]+]] = fir.array_coor %[[V_156]](%[[V_59]]) %arg0 : (!fir.ref<!fir.array<2x!fir.logical<4>>>, !fir.shape<1>, index) -> !fir.ref<!fir.logical<4>>
  ! CHECK:       %[[V_312:[0-9]+]] = fir.load %[[V_311]] : !fir.ref<!fir.logical<4>>
  ! CHECK:       %[[V_313:[0-9]+]] = fir.coordinate_of %[[V_310]], %[[V_82]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>, !fir.field) -> !fir.ref<i8>
  ! CHECK:       %[[V_314:[0-9]+]] = fir.load %[[V_313]] : !fir.ref<i8>
  ! CHECK:       %[[V_315:[0-9]+]] = fir.convert %[[V_314]] : (i8) -> i32
  ! CHECK:       %[[V_316:[0-9]+]] = fir.call @_FortranAMapException(%[[V_315]]) fastmath<contract> : (i32) -> i32
  ! CHECK:       %[[V_317:[0-9]+]] = fir.convert %[[V_312]] : (!fir.logical<4>) -> i1
  ! CHECK:       fir.if %[[V_317]] {
  ! CHECK:         %[[V_318:[0-9]+]] = fir.call @feraiseexcept(%[[V_316]]) fastmath<contract> : (i32) -> i32
  ! CHECK:       } else {
  ! CHECK:         %[[V_318:[0-9]+]] = fir.call @feclearexcept(%[[V_316]]) fastmath<contract> : (i32) -> i32
  ! CHECK:       }
  ! CHECK:     }
  call ieee_set_flag([ieee_invalid, ieee_overflow], [.false., .true.])

  ! CHECK:     %[[V_157:[0-9]+]] = fir.declare %[[V_142]](%[[V_59]]) {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro.2x_QM__fortran_builtinsT__builtin_ieee_flag_type.2"} : (!fir.ref<!fir.array<2x!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>>, !fir.shape<1>) -> !fir.ref<!fir.array<2x!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>>
  ! CHECK:     fir.do_loop %arg0 = %c1{{.*}} to %c2{{.*}} step %c1{{.*}} {
  ! CHECK:       %[[V_310:[0-9]+]] = fir.array_coor %[[V_157]](%[[V_59]]) %arg0 : (!fir.ref<!fir.array<2x!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>>, !fir.shape<1>, index) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>
  ! CHECK:       %[[V_311:[0-9]+]] = fir.array_coor %[[V_60]](%[[V_59]]) %arg0 : (!fir.ref<!fir.array<2x!fir.logical<4>>>, !fir.shape<1>, index) -> !fir.ref<!fir.logical<4>>
  ! CHECK:       %[[V_312:[0-9]+]] = fir.coordinate_of %[[V_310]], %[[V_82]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>, !fir.field) -> !fir.ref<i8>
  ! CHECK:       %[[V_313:[0-9]+]] = fir.load %[[V_312]] : !fir.ref<i8>
  ! CHECK:       %[[V_314:[0-9]+]] = fir.convert %[[V_313]] : (i8) -> i32
  ! CHECK:       %[[V_315:[0-9]+]] = fir.call @_FortranAMapException(%[[V_314]]) fastmath<contract> : (i32) -> i32
  ! CHECK:       %[[V_316:[0-9]+]] = fir.call @fetestexcept(%[[V_315]]) fastmath<contract> : (i32) -> i32
  ! CHECK:       %[[V_317:[0-9]+]] = arith.cmpi ne, %[[V_316]], %c0{{.*}} : i32
  ! CHECK:       %[[V_318:[0-9]+]] = fir.convert %[[V_317]] : (i1) -> !fir.logical<4>
  ! CHECK:       fir.store %[[V_318]] to %[[V_311]] : !fir.ref<!fir.logical<4>>
  ! CHECK:     }
  call ieee_get_flag([ieee_overflow, ieee_invalid], v2)

  ! CHECK:     %[[V_158:[0-9]+]] = fir.call @_FortranAioBeginExternalListOutput
  print*, '[overflow[T], invalid[F]]: ', v2

  ! CHECK:     %[[V_165:[0-9]+]] = fir.address_of(@_QQro.3x_QM__fortran_builtinsT__builtin_ieee_flag_type.4) : !fir.ref<!fir.array<3x!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>>
  ! CHECK:     %[[V_166:[0-9]+]] = fir.declare %[[V_165]](%[[V_54]]) {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro.3x_QM__fortran_builtinsT__builtin_ieee_flag_type.4"} : (!fir.ref<!fir.array<3x!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>>, !fir.shape<1>) -> !fir.ref<!fir.array<3x!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>>
  ! CHECK:     fir.do_loop %arg0 = %c1{{.*}} to %c3{{.*}} step %c1{{.*}} {
  ! CHECK:       %[[V_310:[0-9]+]] = fir.array_coor %[[V_166]](%[[V_54]]) %arg0 : (!fir.ref<!fir.array<3x!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>>, !fir.shape<1>, index) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>
  ! CHECK:       %[[V_311:[0-9]+]] = fir.coordinate_of %[[V_310]], %[[V_82]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>, !fir.field) -> !fir.ref<i8>
  ! CHECK:       %[[V_312:[0-9]+]] = fir.load %[[V_311]] : !fir.ref<i8>
  ! CHECK:       %[[V_313:[0-9]+]] = fir.convert %[[V_312]] : (i8) -> i32
  ! CHECK:       %[[V_314:[0-9]+]] = fir.call @_FortranAMapException(%[[V_313]]) fastmath<contract> : (i32) -> i32
  ! CHECK:       fir.if %true{{[_0-9]*}} {
  ! CHECK:         %[[V_315:[0-9]+]] = fir.call @feraiseexcept(%[[V_314]]) fastmath<contract> : (i32) -> i32
  ! CHECK:       } else {
  ! CHECK:         %[[V_315:[0-9]+]] = fir.call @feclearexcept(%[[V_314]]) fastmath<contract> : (i32) -> i32
  ! CHECK:       }
  ! CHECK:     }
  call ieee_set_flag(ieee_usual, .true.)

  ! CHECK:     %[[V_167:[0-9]+]] = fir.declare %[[V_165]](%[[V_54]]) {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro.3x_QM__fortran_builtinsT__builtin_ieee_flag_type.4"} : (!fir.ref<!fir.array<3x!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>>, !fir.shape<1>) -> !fir.ref<!fir.array<3x!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>>
  ! CHECK:     fir.do_loop %arg0 = %c1{{.*}} to %c3{{.*}} step %c1{{.*}} {
  ! CHECK:       %[[V_310:[0-9]+]] = fir.array_coor %[[V_167]](%[[V_54]]) %arg0 : (!fir.ref<!fir.array<3x!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>>, !fir.shape<1>, index) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>
  ! CHECK:       %[[V_311:[0-9]+]] = fir.array_coor %[[V_64]](%[[V_54]]) %arg0 : (!fir.ref<!fir.array<3x!fir.logical<4>>>, !fir.shape<1>, index) -> !fir.ref<!fir.logical<4>>
  ! CHECK:       %[[V_312:[0-9]+]] = fir.coordinate_of %[[V_310]], %[[V_82]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>, !fir.field) -> !fir.ref<i8>
  ! CHECK:       %[[V_313:[0-9]+]] = fir.load %[[V_312]] : !fir.ref<i8>
  ! CHECK:       %[[V_314:[0-9]+]] = fir.convert %[[V_313]] : (i8) -> i32
  ! CHECK:       %[[V_315:[0-9]+]] = fir.call @_FortranAMapException(%[[V_314]]) fastmath<contract> : (i32) -> i32
  ! CHECK:       %[[V_316:[0-9]+]] = fir.call @fetestexcept(%[[V_315]]) fastmath<contract> : (i32) -> i32
  ! CHECK:       %[[V_317:[0-9]+]] = arith.cmpi ne, %[[V_316]], %c0{{.*}} : i32
  ! CHECK:       %[[V_318:[0-9]+]] = fir.convert %[[V_317]] : (i1) -> !fir.logical<4>
  ! CHECK:       fir.store %[[V_318]] to %[[V_311]] : !fir.ref<!fir.logical<4>>
  ! CHECK:     }
  call ieee_get_flag(ieee_usual, v_usual)

  ! CHECK:     %[[V_168:[0-9]+]] = fir.call @_FortranAioBeginExternalListOutput
  print*, '[overflow[T], divide_by_zero[T], invalid[T]]: ', v_usual

  ! CHECK:     %[[V_178:[0-9]+]] = fir.declare %[[V_165]](%[[V_54]]) {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro.3x_QM__fortran_builtinsT__builtin_ieee_flag_type.4"} : (!fir.ref<!fir.array<3x!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>>, !fir.shape<1>) -> !fir.ref<!fir.array<3x!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>>
  ! CHECK:     %[[V_179:[0-9]+]] = fir.address_of(@_QQro.3xl4.5) : !fir.ref<!fir.array<3x!fir.logical<4>>>
  ! CHECK:     %[[V_180:[0-9]+]] = fir.declare %[[V_179]](%[[V_54]]) {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro.3xl4.5"} : (!fir.ref<!fir.array<3x!fir.logical<4>>>, !fir.shape<1>) -> !fir.ref<!fir.array<3x!fir.logical<4>>>
  ! CHECK:     fir.do_loop %arg0 = %c1{{.*}} to %c3{{.*}} step %c1{{.*}} {
  ! CHECK:       %[[V_310:[0-9]+]] = fir.array_coor %[[V_178]](%[[V_54]]) %arg0 : (!fir.ref<!fir.array<3x!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>>, !fir.shape<1>, index) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>
  ! CHECK:       %[[V_311:[0-9]+]] = fir.array_coor %[[V_180]](%[[V_54]]) %arg0 : (!fir.ref<!fir.array<3x!fir.logical<4>>>, !fir.shape<1>, index) -> !fir.ref<!fir.logical<4>>
  ! CHECK:       %[[V_312:[0-9]+]] = fir.load %[[V_311]] : !fir.ref<!fir.logical<4>>
  ! CHECK:       %[[V_313:[0-9]+]] = fir.coordinate_of %[[V_310]], %[[V_82]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>, !fir.field) -> !fir.ref<i8>
  ! CHECK:       %[[V_314:[0-9]+]] = fir.load %[[V_313]] : !fir.ref<i8>
  ! CHECK:       %[[V_315:[0-9]+]] = fir.convert %[[V_314]] : (i8) -> i32
  ! CHECK:       %[[V_316:[0-9]+]] = fir.call @_FortranAMapException(%[[V_315]]) fastmath<contract> : (i32) -> i32
  ! CHECK:       %[[V_317:[0-9]+]] = fir.convert %[[V_312]] : (!fir.logical<4>) -> i1
  ! CHECK:       fir.if %[[V_317]] {
  ! CHECK:         %[[V_318:[0-9]+]] = fir.call @feraiseexcept(%[[V_316]]) fastmath<contract> : (i32) -> i32
  ! CHECK:       } else {
  ! CHECK:         %[[V_318:[0-9]+]] = fir.call @feclearexcept(%[[V_316]]) fastmath<contract> : (i32) -> i32
  ! CHECK:       }
  ! CHECK:     }
  call ieee_set_flag(ieee_usual, [.true., .false., .true.])

  ! CHECK:     %[[V_181:[0-9]+]] = fir.declare %[[V_165]](%[[V_54]]) {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro.3x_QM__fortran_builtinsT__builtin_ieee_flag_type.4"} : (!fir.ref<!fir.array<3x!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>>, !fir.shape<1>) -> !fir.ref<!fir.array<3x!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>>
  ! CHECK:     fir.do_loop %arg0 = %c1{{.*}} to %c3{{.*}} step %c1{{.*}} {
  ! CHECK:       %[[V_310:[0-9]+]] = fir.array_coor %[[V_181]](%[[V_54]]) %arg0 : (!fir.ref<!fir.array<3x!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>>, !fir.shape<1>, index) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>
  ! CHECK:       %[[V_311:[0-9]+]] = fir.array_coor %[[V_64]](%[[V_54]]) %arg0 : (!fir.ref<!fir.array<3x!fir.logical<4>>>, !fir.shape<1>, index) -> !fir.ref<!fir.logical<4>>
  ! CHECK:       %[[V_312:[0-9]+]] = fir.coordinate_of %[[V_310]], %[[V_82]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>, !fir.field) -> !fir.ref<i8>
  ! CHECK:       %[[V_313:[0-9]+]] = fir.load %[[V_312]] : !fir.ref<i8>
  ! CHECK:       %[[V_314:[0-9]+]] = fir.convert %[[V_313]] : (i8) -> i32
  ! CHECK:       %[[V_315:[0-9]+]] = fir.call @_FortranAMapException(%[[V_314]]) fastmath<contract> : (i32) -> i32
  ! CHECK:       %[[V_316:[0-9]+]] = fir.call @fetestexcept(%[[V_315]]) fastmath<contract> : (i32) -> i32
  ! CHECK:       %[[V_317:[0-9]+]] = arith.cmpi ne, %[[V_316]], %c0{{.*}} : i32
  ! CHECK:       %[[V_318:[0-9]+]] = fir.convert %[[V_317]] : (i1) -> !fir.logical<4>
  ! CHECK:       fir.store %[[V_318]] to %[[V_311]] : !fir.ref<!fir.logical<4>>
  ! CHECK:     }
  call ieee_get_flag(ieee_usual, v_usual)

  ! CHECK:     %[[V_182:[0-9]+]] = fir.call @_FortranAioBeginExternalListOutput
  print*, '[overflow[T], divide_by_zero[F], invalid[T]]: ', v_usual

  ! CHECK:     %[[V_189:[0-9]+]] = fir.address_of(@_QQro.5x_QM__fortran_builtinsT__builtin_ieee_flag_type.6) : !fir.ref<!fir.array<5x!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>>
  ! CHECK:     %[[V_190:[0-9]+]] = fir.declare %[[V_189]](%[[V_1]]) {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro.5x_QM__fortran_builtinsT__builtin_ieee_flag_type.6"} : (!fir.ref<!fir.array<5x!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>>, !fir.shape<1>) -> !fir.ref<!fir.array<5x!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>>
  ! CHECK:     fir.do_loop %arg0 = %c1{{.*}} to %c5{{.*}} step %c1{{.*}} {
  ! CHECK:       %[[V_310:[0-9]+]] = fir.array_coor %[[V_190]](%[[V_1]]) %arg0 : (!fir.ref<!fir.array<5x!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>>, !fir.shape<1>, index) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>
  ! CHECK:       %[[V_311:[0-9]+]] = fir.coordinate_of %[[V_310]], %[[V_82]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>, !fir.field) -> !fir.ref<i8>
  ! CHECK:       %[[V_312:[0-9]+]] = fir.load %[[V_311]] : !fir.ref<i8>
  ! CHECK:       %[[V_313:[0-9]+]] = fir.convert %[[V_312]] : (i8) -> i32
  ! CHECK:       %[[V_314:[0-9]+]] = fir.call @_FortranAMapException(%[[V_313]]) fastmath<contract> : (i32) -> i32
  ! CHECK:       fir.if %false{{[_0-9]*}} {
  ! CHECK:         %[[V_315:[0-9]+]] = fir.call @feraiseexcept(%[[V_314]]) fastmath<contract> : (i32) -> i32
  ! CHECK:       } else {
  ! CHECK:         %[[V_315:[0-9]+]] = fir.call @feclearexcept(%[[V_314]]) fastmath<contract> : (i32) -> i32
  ! CHECK:       }
  ! CHECK:     }
  call ieee_set_flag(ieee_all, .false.)

  ! CHECK:     %[[V_191:[0-9]+]] = fir.declare %[[V_189]](%[[V_1]]) {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro.5x_QM__fortran_builtinsT__builtin_ieee_flag_type.6"} : (!fir.ref<!fir.array<5x!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>>, !fir.shape<1>) -> !fir.ref<!fir.array<5x!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>>
  ! CHECK:     fir.do_loop %arg0 = %c1{{.*}} to %c5{{.*}} step %c1{{.*}} {
  ! CHECK:       %[[V_310:[0-9]+]] = fir.array_coor %[[V_191]](%[[V_1]]) %arg0 : (!fir.ref<!fir.array<5x!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>>, !fir.shape<1>, index) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>
  ! CHECK:       %[[V_311:[0-9]+]] = fir.array_coor %[[V_62]](%[[V_1]]) %arg0 : (!fir.ref<!fir.array<5x!fir.logical<4>>>, !fir.shape<1>, index) -> !fir.ref<!fir.logical<4>>
  ! CHECK:       %[[V_312:[0-9]+]] = fir.coordinate_of %[[V_310]], %[[V_82]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>, !fir.field) -> !fir.ref<i8>
  ! CHECK:       %[[V_313:[0-9]+]] = fir.load %[[V_312]] : !fir.ref<i8>
  ! CHECK:       %[[V_314:[0-9]+]] = fir.convert %[[V_313]] : (i8) -> i32
  ! CHECK:       %[[V_315:[0-9]+]] = fir.call @_FortranAMapException(%[[V_314]]) fastmath<contract> : (i32) -> i32
  ! CHECK:       %[[V_316:[0-9]+]] = fir.call @fetestexcept(%[[V_315]]) fastmath<contract> : (i32) -> i32
  ! CHECK:       %[[V_317:[0-9]+]] = arith.cmpi ne, %[[V_316]], %c0{{.*}} : i32
  ! CHECK:       %[[V_318:[0-9]+]] = fir.convert %[[V_317]] : (i1) -> !fir.logical<4>
  ! CHECK:       fir.store %[[V_318]] to %[[V_311]] : !fir.ref<!fir.logical<4>>
  ! CHECK:     }
  call ieee_get_flag(ieee_all, v_all)

  ! CHECK:     %[[V_192:[0-9]+]] = fir.call @_FortranAioBeginExternalListOutput
  print*, 'all[F]: ', v_all

  ! CHECK:     %[[V_202:[0-9]+]] = fir.call @_FortranAioBeginExternalListOutput
  print*

  ! CHECK:     %[[V_204:[0-9]+]] = fir.call @_FortranAioBeginExternalListOutput
  print*, 'Halting'

  ! CHECK:     %[[V_211:[0-9]+]] = fir.call @_FortranAioBeginExternalListOutput
  ! CHECK:     %[[V_220:[0-9]+]] = fir.call @_FortranAioOutputLogical(%[[V_211]]
  print*, 'support invalid: ', ieee_support_halting(ieee_invalid)

  ! CHECK:     %[[V_222:[0-9]+]] = fir.declare %[[V_80]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro._QM__fortran_builtinsT__builtin_ieee_flag_type.0"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>
  ! CHECK:     %[[V_223:[0-9]+]] = fir.coordinate_of %[[V_222]], %[[V_82]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>, !fir.field) -> !fir.ref<i8>
  ! CHECK:     %[[V_224:[0-9]+]] = fir.load %[[V_223]] : !fir.ref<i8>
  ! CHECK:     %[[V_225:[0-9]+]] = fir.convert %[[V_224]] : (i8) -> i32
  ! CHECK:     %[[V_226:[0-9]+]] = fir.call @_FortranAMapException(%[[V_225]]) fastmath<contract> : (i32) -> i32
  ! CHECK:     fir.if %false{{[_0-9]*}} {
  ! CHECK:       %[[V_310:[0-9]+]] = fir.call @feenableexcept(%[[V_226]]) fastmath<contract> : (i32) -> i32
  ! CHECK:     } else {
  ! CHECK:       %[[V_310:[0-9]+]] = fir.call @fedisableexcept(%[[V_226]]) fastmath<contract> : (i32) -> i32
  ! CHECK:     }
  call ieee_set_halting_mode(ieee_invalid, .false.)

  ! CHECK:     %[[V_227:[0-9]+]] = fir.declare %[[V_80]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro._QM__fortran_builtinsT__builtin_ieee_flag_type.0"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>
  ! CHECK:     %[[V_228:[0-9]+]] = fir.coordinate_of %[[V_227]], %[[V_82]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>, !fir.field) -> !fir.ref<i8>
  ! CHECK:     %[[V_229:[0-9]+]] = fir.load %[[V_228]] : !fir.ref<i8>
  ! CHECK:     %[[V_230:[0-9]+]] = fir.call @fegetexcept() fastmath<contract> : () -> i32
  ! CHECK:     %[[V_231:[0-9]+]] = fir.convert %[[V_229]] : (i8) -> i32
  ! CHECK:     %[[V_232:[0-9]+]] = fir.call @_FortranAMapException(%[[V_231]]) fastmath<contract> : (i32) -> i32
  ! CHECK:     %[[V_233:[0-9]+]] = arith.andi %[[V_230]], %[[V_232]] : i32
  ! CHECK:     %[[V_234:[0-9]+]] = arith.cmpi ne, %[[V_233]], %c0{{.*}} : i32
  ! CHECK:     %[[V_235:[0-9]+]] = fir.convert %[[V_234]] : (i1) -> !fir.logical<4>
  ! CHECK:     fir.store %[[V_235]] to %[[V_57]] : !fir.ref<!fir.logical<4>>
  call ieee_get_halting_mode(ieee_invalid, v)

  ! CHECK:     %[[V_236:[0-9]+]] = fir.call @_FortranAioBeginExternalListOutput
  print*, 'invalid[F]: ', v

  ! CHECK:     %[[V_244:[0-9]+]] = fir.declare %[[V_80]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro._QM__fortran_builtinsT__builtin_ieee_flag_type.0"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>
  ! CHECK:     %[[V_245:[0-9]+]] = fir.coordinate_of %[[V_244]], %[[V_82]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>, !fir.field) -> !fir.ref<i8>
  ! CHECK:     %[[V_246:[0-9]+]] = fir.load %[[V_245]] : !fir.ref<i8>
  ! CHECK:     %[[V_247:[0-9]+]] = fir.convert %[[V_246]] : (i8) -> i32
  ! CHECK:     %[[V_248:[0-9]+]] = fir.call @_FortranAMapException(%[[V_247]]) fastmath<contract> : (i32) -> i32
  ! CHECK:     fir.if %true{{[_0-9]*}} {
  ! CHECK:       %[[V_310:[0-9]+]] = fir.call @feenableexcept(%[[V_248]]) fastmath<contract> : (i32) -> i32
  ! CHECK:     } else {
  ! CHECK:       %[[V_310:[0-9]+]] = fir.call @fedisableexcept(%[[V_248]]) fastmath<contract> : (i32) -> i32
  ! CHECK:     }
  call ieee_set_halting_mode(ieee_invalid, .true.)

  ! CHECK:     %[[V_249:[0-9]+]] = fir.declare %[[V_80]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro._QM__fortran_builtinsT__builtin_ieee_flag_type.0"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>
  ! CHECK:     %[[V_250:[0-9]+]] = fir.coordinate_of %[[V_249]], %[[V_82]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>, !fir.field) -> !fir.ref<i8>
  ! CHECK:     %[[V_251:[0-9]+]] = fir.load %[[V_250]] : !fir.ref<i8>
  ! CHECK:     %[[V_252:[0-9]+]] = fir.call @fegetexcept() fastmath<contract> : () -> i32
  ! CHECK:     %[[V_253:[0-9]+]] = fir.convert %[[V_251]] : (i8) -> i32
  ! CHECK:     %[[V_254:[0-9]+]] = fir.call @_FortranAMapException(%[[V_253]]) fastmath<contract> : (i32) -> i32
  ! CHECK:     %[[V_255:[0-9]+]] = arith.andi %[[V_252]], %[[V_254]] : i32
  ! CHECK:     %[[V_256:[0-9]+]] = arith.cmpi ne, %[[V_255]], %c0{{.*}} : i32
  ! CHECK:     %[[V_257:[0-9]+]] = fir.convert %[[V_256]] : (i1) -> !fir.logical<4>
  ! CHECK:     fir.store %[[V_257]] to %[[V_57]] : !fir.ref<!fir.logical<4>>
  call ieee_get_halting_mode(ieee_invalid, v)

  ! CHECK:     %[[V_258:[0-9]+]] = fir.call @_FortranAioBeginExternalListOutput
  print*, 'invalid[T]: ', v

  ! CHECK:     %[[V_266:[0-9]+]] = fir.declare %[[V_140]](%[[V_59]]) {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro.2x_QM__fortran_builtinsT__builtin_ieee_flag_type.1"} : (!fir.ref<!fir.array<2x!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>>, !fir.shape<1>) -> !fir.ref<!fir.array<2x!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>>
  ! CHECK:     fir.do_loop %arg0 = %c1{{.*}} to %c2{{.*}} step %c1{{.*}} {
  ! CHECK:       %[[V_310:[0-9]+]] = fir.array_coor %[[V_266]](%[[V_59]]) %arg0 : (!fir.ref<!fir.array<2x!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>>, !fir.shape<1>, index) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>
  ! CHECK:       %[[V_311:[0-9]+]] = fir.coordinate_of %[[V_310]], %[[V_82]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>, !fir.field) -> !fir.ref<i8>
  ! CHECK:       %[[V_312:[0-9]+]] = fir.load %[[V_311]] : !fir.ref<i8>
  ! CHECK:       %[[V_313:[0-9]+]] = fir.convert %[[V_312]] : (i8) -> i32
  ! CHECK:       %[[V_314:[0-9]+]] = fir.call @_FortranAMapException(%[[V_313]]) fastmath<contract> : (i32) -> i32
  ! CHECK:       fir.if %false{{[_0-9]*}} {
  ! CHECK:         %[[V_315:[0-9]+]] = fir.call @feenableexcept(%[[V_314]]) fastmath<contract> : (i32) -> i32
  ! CHECK:       } else {
  ! CHECK:         %[[V_315:[0-9]+]] = fir.call @fedisableexcept(%[[V_314]]) fastmath<contract> : (i32) -> i32
  ! CHECK:       }
  ! CHECK:     }
  call ieee_set_halting_mode([ieee_invalid, ieee_overflow], .false.)

  ! CHECK:     %[[V_267:[0-9]+]] = fir.declare %[[V_142]](%[[V_59]]) {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro.2x_QM__fortran_builtinsT__builtin_ieee_flag_type.2"} : (!fir.ref<!fir.array<2x!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>>, !fir.shape<1>) -> !fir.ref<!fir.array<2x!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>>
  ! CHECK:     fir.do_loop %arg0 = %c1{{.*}} to %c2{{.*}} step %c1{{.*}} {
  ! CHECK:       %[[V_310:[0-9]+]] = fir.array_coor %[[V_267]](%[[V_59]]) %arg0 : (!fir.ref<!fir.array<2x!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>>, !fir.shape<1>, index) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>
  ! CHECK:       %[[V_311:[0-9]+]] = fir.array_coor %[[V_60]](%[[V_59]]) %arg0 : (!fir.ref<!fir.array<2x!fir.logical<4>>>, !fir.shape<1>, index) -> !fir.ref<!fir.logical<4>>
  ! CHECK:       %[[V_312:[0-9]+]] = fir.coordinate_of %[[V_310]], %[[V_82]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>, !fir.field) -> !fir.ref<i8>
  ! CHECK:       %[[V_313:[0-9]+]] = fir.load %[[V_312]] : !fir.ref<i8>
  ! CHECK:       %[[V_314:[0-9]+]] = fir.call @fegetexcept() fastmath<contract> : () -> i32
  ! CHECK:       %[[V_315:[0-9]+]] = fir.convert %[[V_313]] : (i8) -> i32
  ! CHECK:       %[[V_316:[0-9]+]] = fir.call @_FortranAMapException(%[[V_315]]) fastmath<contract> : (i32) -> i32
  ! CHECK:       %[[V_317:[0-9]+]] = arith.andi %[[V_314]], %[[V_316]] : i32
  ! CHECK:       %[[V_318:[0-9]+]] = arith.cmpi ne, %[[V_317]], %c0{{.*}} : i32
  ! CHECK:       %[[V_319:[0-9]+]] = fir.convert %[[V_318]] : (i1) -> !fir.logical<4>
  ! CHECK:       fir.store %[[V_319]] to %[[V_311]] : !fir.ref<!fir.logical<4>>
  ! CHECK:     }
  call ieee_get_halting_mode([ieee_overflow, ieee_invalid], v2)

  ! CHECK:     %[[V_268:[0-9]+]] = fir.call @_FortranAioBeginExternalListOutput
  print*, '[overflow[F], invalid[F]]: ', v2

  ! CHECK:     %[[V_274:[0-9]+]] = fir.declare %[[V_140]](%[[V_59]]) {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro.2x_QM__fortran_builtinsT__builtin_ieee_flag_type.1"} : (!fir.ref<!fir.array<2x!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>>, !fir.shape<1>) -> !fir.ref<!fir.array<2x!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>>
  ! CHECK:     %[[V_275:[0-9]+]] = fir.declare %[[V_155]](%[[V_59]]) {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro.2xl4.3"} : (!fir.ref<!fir.array<2x!fir.logical<4>>>, !fir.shape<1>) -> !fir.ref<!fir.array<2x!fir.logical<4>>>
  ! CHECK:     fir.do_loop %arg0 = %c1{{.*}} to %c2{{.*}} step %c1{{.*}} {
  ! CHECK:       %[[V_310:[0-9]+]] = fir.array_coor %[[V_274]](%[[V_59]]) %arg0 : (!fir.ref<!fir.array<2x!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>>, !fir.shape<1>, index) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>
  ! CHECK:       %[[V_311:[0-9]+]] = fir.array_coor %[[V_275]](%[[V_59]]) %arg0 : (!fir.ref<!fir.array<2x!fir.logical<4>>>, !fir.shape<1>, index) -> !fir.ref<!fir.logical<4>>
  ! CHECK:       %[[V_312:[0-9]+]] = fir.load %[[V_311]] : !fir.ref<!fir.logical<4>>
  ! CHECK:       %[[V_313:[0-9]+]] = fir.coordinate_of %[[V_310]], %[[V_82]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>, !fir.field) -> !fir.ref<i8>
  ! CHECK:       %[[V_314:[0-9]+]] = fir.load %[[V_313]] : !fir.ref<i8>
  ! CHECK:       %[[V_315:[0-9]+]] = fir.convert %[[V_314]] : (i8) -> i32
  ! CHECK:       %[[V_316:[0-9]+]] = fir.call @_FortranAMapException(%[[V_315]]) fastmath<contract> : (i32) -> i32
  ! CHECK:       %[[V_317:[0-9]+]] = fir.convert %[[V_312]] : (!fir.logical<4>) -> i1
  ! CHECK:       fir.if %[[V_317]] {
  ! CHECK:         %[[V_318:[0-9]+]] = fir.call @feenableexcept(%[[V_316]]) fastmath<contract> : (i32) -> i32
  ! CHECK:       } else {
  ! CHECK:         %[[V_318:[0-9]+]] = fir.call @fedisableexcept(%[[V_316]]) fastmath<contract> : (i32) -> i32
  ! CHECK:       }
  ! CHECK:     }
  call ieee_set_halting_mode([ieee_invalid, ieee_overflow], [.false., .true.])

  ! CHECK:     %[[V_276:[0-9]+]] = fir.declare %[[V_142]](%[[V_59]]) {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro.2x_QM__fortran_builtinsT__builtin_ieee_flag_type.2"} : (!fir.ref<!fir.array<2x!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>>, !fir.shape<1>) -> !fir.ref<!fir.array<2x!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>>
  ! CHECK:     fir.do_loop %arg0 = %c1{{.*}} to %c2{{.*}} step %c1{{.*}} {
  ! CHECK:       %[[V_310:[0-9]+]] = fir.array_coor %[[V_276]](%[[V_59]]) %arg0 : (!fir.ref<!fir.array<2x!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>>, !fir.shape<1>, index) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>
  ! CHECK:       %[[V_311:[0-9]+]] = fir.array_coor %[[V_60]](%[[V_59]]) %arg0 : (!fir.ref<!fir.array<2x!fir.logical<4>>>, !fir.shape<1>, index) -> !fir.ref<!fir.logical<4>>
  ! CHECK:       %[[V_312:[0-9]+]] = fir.coordinate_of %[[V_310]], %[[V_82]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>, !fir.field) -> !fir.ref<i8>
  ! CHECK:       %[[V_313:[0-9]+]] = fir.load %[[V_312]] : !fir.ref<i8>
  ! CHECK:       %[[V_314:[0-9]+]] = fir.call @fegetexcept() fastmath<contract> : () -> i32
  ! CHECK:       %[[V_315:[0-9]+]] = fir.convert %[[V_313]] : (i8) -> i32
  ! CHECK:       %[[V_316:[0-9]+]] = fir.call @_FortranAMapException(%[[V_315]]) fastmath<contract> : (i32) -> i32
  ! CHECK:       %[[V_317:[0-9]+]] = arith.andi %[[V_314]], %[[V_316]] : i32
  ! CHECK:       %[[V_318:[0-9]+]] = arith.cmpi ne, %[[V_317]], %c0{{.*}} : i32
  ! CHECK:       %[[V_319:[0-9]+]] = fir.convert %[[V_318]] : (i1) -> !fir.logical<4>
  ! CHECK:       fir.store %[[V_319]] to %[[V_311]] : !fir.ref<!fir.logical<4>>
  ! CHECK:     }
  call ieee_get_halting_mode([ieee_overflow, ieee_invalid], v2)

  ! CHECK:     %[[V_277:[0-9]+]] = fir.call @_FortranAioBeginExternalListOutput
  print*, '[overflow[T], invalid[F]]: ', v2

  ! CHECK:     %[[V_283:[0-9]+]] = fir.declare %[[V_165]](%[[V_54]]) {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro.3x_QM__fortran_builtinsT__builtin_ieee_flag_type.4"} : (!fir.ref<!fir.array<3x!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>>, !fir.shape<1>) -> !fir.ref<!fir.array<3x!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>>
  ! CHECK:     fir.do_loop %arg0 = %c1{{.*}} to %c3{{.*}} step %c1{{.*}} {
  ! CHECK:       %[[V_310:[0-9]+]] = fir.array_coor %[[V_283]](%[[V_54]]) %arg0 : (!fir.ref<!fir.array<3x!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>>, !fir.shape<1>, index) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>
  ! CHECK:       %[[V_311:[0-9]+]] = fir.coordinate_of %[[V_310]], %[[V_82]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>, !fir.field) -> !fir.ref<i8>
  ! CHECK:       %[[V_312:[0-9]+]] = fir.load %[[V_311]] : !fir.ref<i8>
  ! CHECK:       %[[V_313:[0-9]+]] = fir.convert %[[V_312]] : (i8) -> i32
  ! CHECK:       %[[V_314:[0-9]+]] = fir.call @_FortranAMapException(%[[V_313]]) fastmath<contract> : (i32) -> i32
  ! CHECK:       fir.if %true{{[_0-9]*}} {
  ! CHECK:         %[[V_315:[0-9]+]] = fir.call @feenableexcept(%[[V_314]]) fastmath<contract> : (i32) -> i32
  ! CHECK:       } else {
  ! CHECK:         %[[V_315:[0-9]+]] = fir.call @fedisableexcept(%[[V_314]]) fastmath<contract> : (i32) -> i32
  ! CHECK:       }
  ! CHECK:     }
  call ieee_set_halting_mode(ieee_usual, .true.)

  ! CHECK:     %[[V_284:[0-9]+]] = fir.declare %[[V_165]](%[[V_54]]) {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro.3x_QM__fortran_builtinsT__builtin_ieee_flag_type.4"} : (!fir.ref<!fir.array<3x!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>>, !fir.shape<1>) -> !fir.ref<!fir.array<3x!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>>
  ! CHECK:     fir.do_loop %arg0 = %c1{{.*}} to %c3{{.*}} step %c1{{.*}} {
  ! CHECK:       %[[V_310:[0-9]+]] = fir.array_coor %[[V_284]](%[[V_54]]) %arg0 : (!fir.ref<!fir.array<3x!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>>, !fir.shape<1>, index) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>
  ! CHECK:       %[[V_311:[0-9]+]] = fir.array_coor %[[V_64]](%[[V_54]]) %arg0 : (!fir.ref<!fir.array<3x!fir.logical<4>>>, !fir.shape<1>, index) -> !fir.ref<!fir.logical<4>>
  ! CHECK:       %[[V_312:[0-9]+]] = fir.coordinate_of %[[V_310]], %[[V_82]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>, !fir.field) -> !fir.ref<i8>
  ! CHECK:       %[[V_313:[0-9]+]] = fir.load %[[V_312]] : !fir.ref<i8>
  ! CHECK:       %[[V_314:[0-9]+]] = fir.call @fegetexcept() fastmath<contract> : () -> i32
  ! CHECK:       %[[V_315:[0-9]+]] = fir.convert %[[V_313]] : (i8) -> i32
  ! CHECK:       %[[V_316:[0-9]+]] = fir.call @_FortranAMapException(%[[V_315]]) fastmath<contract> : (i32) -> i32
  ! CHECK:       %[[V_317:[0-9]+]] = arith.andi %[[V_314]], %[[V_316]] : i32
  ! CHECK:       %[[V_318:[0-9]+]] = arith.cmpi ne, %[[V_317]], %c0{{.*}} : i32
  ! CHECK:       %[[V_319:[0-9]+]] = fir.convert %[[V_318]] : (i1) -> !fir.logical<4>
  ! CHECK:       fir.store %[[V_319]] to %[[V_311]] : !fir.ref<!fir.logical<4>>
  ! CHECK:     }
  call ieee_get_halting_mode(ieee_usual, v_usual)

  ! CHECK:     %[[V_285:[0-9]+]] = fir.call @_FortranAioBeginExternalListOutput
  print*, '[overflow[T], divide_by_zero[T], invalid[T]]: ', v_usual

  ! CHECK:     %[[V_291:[0-9]+]] = fir.declare %[[V_165]](%[[V_54]]) {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro.3x_QM__fortran_builtinsT__builtin_ieee_flag_type.4"} : (!fir.ref<!fir.array<3x!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>>, !fir.shape<1>) -> !fir.ref<!fir.array<3x!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>>
  ! CHECK:     %[[V_292:[0-9]+]] = fir.declare %[[V_179]](%[[V_54]]) {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro.3xl4.5"} : (!fir.ref<!fir.array<3x!fir.logical<4>>>, !fir.shape<1>) -> !fir.ref<!fir.array<3x!fir.logical<4>>>
  ! CHECK:     fir.do_loop %arg0 = %c1{{.*}} to %c3{{.*}} step %c1{{.*}} {
  ! CHECK:       %[[V_310:[0-9]+]] = fir.array_coor %[[V_291]](%[[V_54]]) %arg0 : (!fir.ref<!fir.array<3x!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>>, !fir.shape<1>, index) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>
  ! CHECK:       %[[V_311:[0-9]+]] = fir.array_coor %[[V_292]](%[[V_54]]) %arg0 : (!fir.ref<!fir.array<3x!fir.logical<4>>>, !fir.shape<1>, index) -> !fir.ref<!fir.logical<4>>
  ! CHECK:       %[[V_312:[0-9]+]] = fir.load %[[V_311]] : !fir.ref<!fir.logical<4>>
  ! CHECK:       %[[V_313:[0-9]+]] = fir.coordinate_of %[[V_310]], %[[V_82]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>, !fir.field) -> !fir.ref<i8>
  ! CHECK:       %[[V_314:[0-9]+]] = fir.load %[[V_313]] : !fir.ref<i8>
  ! CHECK:       %[[V_315:[0-9]+]] = fir.convert %[[V_314]] : (i8) -> i32
  ! CHECK:       %[[V_316:[0-9]+]] = fir.call @_FortranAMapException(%[[V_315]]) fastmath<contract> : (i32) -> i32
  ! CHECK:       %[[V_317:[0-9]+]] = fir.convert %[[V_312]] : (!fir.logical<4>) -> i1
  ! CHECK:       fir.if %[[V_317]] {
  ! CHECK:         %[[V_318:[0-9]+]] = fir.call @feenableexcept(%[[V_316]]) fastmath<contract> : (i32) -> i32
  ! CHECK:       } else {
  ! CHECK:         %[[V_318:[0-9]+]] = fir.call @fedisableexcept(%[[V_316]]) fastmath<contract> : (i32) -> i32
  ! CHECK:       }
  ! CHECK:     }
  call ieee_set_halting_mode(ieee_usual, [.true., .false., .true.])

  ! CHECK:     %[[V_293:[0-9]+]] = fir.declare %[[V_165]](%[[V_54]]) {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro.3x_QM__fortran_builtinsT__builtin_ieee_flag_type.4"} : (!fir.ref<!fir.array<3x!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>>, !fir.shape<1>) -> !fir.ref<!fir.array<3x!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>>
  ! CHECK:     fir.do_loop %arg0 = %c1{{.*}} to %c3{{.*}} step %c1{{.*}} {
  ! CHECK:       %[[V_310:[0-9]+]] = fir.array_coor %[[V_293]](%[[V_54]]) %arg0 : (!fir.ref<!fir.array<3x!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>>, !fir.shape<1>, index) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>
  ! CHECK:       %[[V_311:[0-9]+]] = fir.array_coor %[[V_64]](%[[V_54]]) %arg0 : (!fir.ref<!fir.array<3x!fir.logical<4>>>, !fir.shape<1>, index) -> !fir.ref<!fir.logical<4>>
  ! CHECK:       %[[V_312:[0-9]+]] = fir.coordinate_of %[[V_310]], %[[V_82]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>, !fir.field) -> !fir.ref<i8>
  ! CHECK:       %[[V_313:[0-9]+]] = fir.load %[[V_312]] : !fir.ref<i8>
  ! CHECK:       %[[V_314:[0-9]+]] = fir.call @fegetexcept() fastmath<contract> : () -> i32
  ! CHECK:       %[[V_315:[0-9]+]] = fir.convert %[[V_313]] : (i8) -> i32
  ! CHECK:       %[[V_316:[0-9]+]] = fir.call @_FortranAMapException(%[[V_315]]) fastmath<contract> : (i32) -> i32
  ! CHECK:       %[[V_317:[0-9]+]] = arith.andi %[[V_314]], %[[V_316]] : i32
  ! CHECK:       %[[V_318:[0-9]+]] = arith.cmpi ne, %[[V_317]], %c0{{.*}} : i32
  ! CHECK:       %[[V_319:[0-9]+]] = fir.convert %[[V_318]] : (i1) -> !fir.logical<4>
  ! CHECK:       fir.store %[[V_319]] to %[[V_311]] : !fir.ref<!fir.logical<4>>
  ! CHECK:     }
  call ieee_get_halting_mode(ieee_usual, v_usual)

  ! CHECK:     %[[V_294:[0-9]+]] = fir.call @_FortranAioBeginExternalListOutput
  print*, '[overflow[T], divide_by_zero[F], invalid[T]]: ', v_usual

  ! CHECK:     %[[V_300:[0-9]+]] = fir.declare %[[V_189]](%[[V_1]]) {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro.5x_QM__fortran_builtinsT__builtin_ieee_flag_type.6"} : (!fir.ref<!fir.array<5x!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>>, !fir.shape<1>) -> !fir.ref<!fir.array<5x!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>>
  ! CHECK:     fir.do_loop %arg0 = %c1{{.*}} to %c5{{.*}} step %c1{{.*}} {
  ! CHECK:       %[[V_310:[0-9]+]] = fir.array_coor %[[V_300]](%[[V_1]]) %arg0 : (!fir.ref<!fir.array<5x!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>>, !fir.shape<1>, index) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>
  ! CHECK:       %[[V_311:[0-9]+]] = fir.coordinate_of %[[V_310]], %[[V_82]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>, !fir.field) -> !fir.ref<i8>
  ! CHECK:       %[[V_312:[0-9]+]] = fir.load %[[V_311]] : !fir.ref<i8>
  ! CHECK:       %[[V_313:[0-9]+]] = fir.convert %[[V_312]] : (i8) -> i32
  ! CHECK:       %[[V_314:[0-9]+]] = fir.call @_FortranAMapException(%[[V_313]]) fastmath<contract> : (i32) -> i32
  ! CHECK:       fir.if %true{{[_0-9]*}} {
  ! CHECK:         %[[V_315:[0-9]+]] = fir.call @feenableexcept(%[[V_314]]) fastmath<contract> : (i32) -> i32
  ! CHECK:       } else {
  ! CHECK:         %[[V_315:[0-9]+]] = fir.call @fedisableexcept(%[[V_314]]) fastmath<contract> : (i32) -> i32
  ! CHECK:       }
  ! CHECK:     }
  call ieee_set_halting_mode(ieee_all, .true.)

  ! CHECK:     %[[V_301:[0-9]+]] = fir.declare %[[V_189]](%[[V_1]]) {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro.5x_QM__fortran_builtinsT__builtin_ieee_flag_type.6"} : (!fir.ref<!fir.array<5x!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>>, !fir.shape<1>) -> !fir.ref<!fir.array<5x!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>>
  ! CHECK:     fir.do_loop %arg0 = %c1{{.*}} to %c5{{.*}} step %c1{{.*}} {
  ! CHECK:       %[[V_310:[0-9]+]] = fir.array_coor %[[V_301]](%[[V_1]]) %arg0 : (!fir.ref<!fir.array<5x!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>>, !fir.shape<1>, index) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>
  ! CHECK:       %[[V_311:[0-9]+]] = fir.array_coor %[[V_62]](%[[V_1]]) %arg0 : (!fir.ref<!fir.array<5x!fir.logical<4>>>, !fir.shape<1>, index) -> !fir.ref<!fir.logical<4>>
  ! CHECK:       %[[V_312:[0-9]+]] = fir.coordinate_of %[[V_310]], %[[V_82]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_flag_type{_QM__fortran_builtinsT__builtin_ieee_flag_type.flag:i8}>>, !fir.field) -> !fir.ref<i8>
  ! CHECK:       %[[V_313:[0-9]+]] = fir.load %[[V_312]] : !fir.ref<i8>
  ! CHECK:       %[[V_314:[0-9]+]] = fir.call @fegetexcept() fastmath<contract> : () -> i32
  ! CHECK:       %[[V_315:[0-9]+]] = fir.convert %[[V_313]] : (i8) -> i32
  ! CHECK:       %[[V_316:[0-9]+]] = fir.call @_FortranAMapException(%[[V_315]]) fastmath<contract> : (i32) -> i32
  ! CHECK:       %[[V_317:[0-9]+]] = arith.andi %[[V_314]], %[[V_316]] : i32
  ! CHECK:       %[[V_318:[0-9]+]] = arith.cmpi ne, %[[V_317]], %c0{{.*}} : i32
  ! CHECK:       %[[V_319:[0-9]+]] = fir.convert %[[V_318]] : (i1) -> !fir.logical<4>
  ! CHECK:       fir.store %[[V_319]] to %[[V_311]] : !fir.ref<!fir.logical<4>>
  ! CHECK:     }
  call ieee_get_halting_mode(ieee_all, v_all)

  ! CHECK:     %[[V_302:[0-9]+]] = fir.call @_FortranAioBeginExternalListOutput
  print*, 'all[T]: ', v_all

  stop
  end
