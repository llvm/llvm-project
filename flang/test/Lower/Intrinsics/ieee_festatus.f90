! RUN: bbc -emit-fir -o - %s | FileCheck %s

! CHECK-LABEL: c.func @_QQmain
program s
  use ieee_arithmetic

  ! CHECK:     %[[V_0:[0-9]+]] = fir.address_of(@_QM__fortran_ieee_exceptionsECieee_all) : !fir.ref<!fir.array<5x!fir.type<_QM__fortran_ieee_exceptionsTieee_flag_type{_QM__fortran_ieee_exceptionsTieee_flag_type.flag:i8}>>>
  ! CHECK:     %[[V_1:[0-9]+]] = fir.shape %c5{{.*}} : (index) -> !fir.shape<1>
  ! CHECK:     %[[V_2:[0-9]+]] = fir.declare %[[V_0]](%[[V_1]]) {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QM__fortran_ieee_exceptionsECieee_all"} : (!fir.ref<!fir.array<5x!fir.type<_QM__fortran_ieee_exceptionsTieee_flag_type{_QM__fortran_ieee_exceptionsTieee_flag_type.flag:i8}>>>, !fir.shape<1>) -> !fir.ref<!fir.array<5x!fir.type<_QM__fortran_ieee_exceptionsTieee_flag_type{_QM__fortran_ieee_exceptionsTieee_flag_type.flag:i8}>>>
  ! CHECK:     %[[V_53:[0-9]+]] = fir.address_of(@_QM__fortran_ieee_exceptionsECieee_usual) : !fir.ref<!fir.array<3x!fir.type<_QM__fortran_ieee_exceptionsTieee_flag_type{_QM__fortran_ieee_exceptionsTieee_flag_type.flag:i8}>>>
  ! CHECK:     %[[V_54:[0-9]+]] = fir.shape %c3{{.*}} : (index) -> !fir.shape<1>
  ! CHECK:     %[[V_55:[0-9]+]] = fir.declare %[[V_53]](%[[V_54]]) {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QM__fortran_ieee_exceptionsECieee_usual"} : (!fir.ref<!fir.array<3x!fir.type<_QM__fortran_ieee_exceptionsTieee_flag_type{_QM__fortran_ieee_exceptionsTieee_flag_type.flag:i8}>>>, !fir.shape<1>) -> !fir.ref<!fir.array<3x!fir.type<_QM__fortran_ieee_exceptionsTieee_flag_type{_QM__fortran_ieee_exceptionsTieee_flag_type.flag:i8}>>>
  use ieee_exceptions

  ! CHECK:     %[[V_56:[0-9]+]] = fir.alloca !fir.type<_QM__fortran_ieee_exceptionsTieee_status_type{_QM__fortran_ieee_exceptionsTieee_status_type.__data:!fir.array<8xi32>}> {bindc_name = "status", uniq_name = "_QFEstatus"}
  ! CHECK:     %[[V_57:[0-9]+]] = fir.declare %[[V_56]] {uniq_name = "_QFEstatus"} : (!fir.ref<!fir.type<_QM__fortran_ieee_exceptionsTieee_status_type{_QM__fortran_ieee_exceptionsTieee_status_type.__data:!fir.array<8xi32>}>>) -> !fir.ref<!fir.type<_QM__fortran_ieee_exceptionsTieee_status_type{_QM__fortran_ieee_exceptionsTieee_status_type.__data:!fir.array<8xi32>}>>
  type(ieee_status_type) :: status

  ! CHECK:     %[[V_58:[0-9]+]] = fir.alloca !fir.array<5x!fir.logical<4>> {bindc_name = "v", uniq_name = "_QFEv"}
  ! CHECK:     %[[V_59:[0-9]+]] = fir.declare %[[V_58]](%[[V_1]]) {uniq_name = "_QFEv"} : (!fir.ref<!fir.array<5x!fir.logical<4>>>, !fir.shape<1>) -> !fir.ref<!fir.array<5x!fir.logical<4>>>
  logical :: v(size(ieee_all))

  ! CHECK:     %[[V_60:[0-9]+]] = fir.address_of(@_QQro.5x_QM__fortran_ieee_exceptionsTieee_flag_type.0) : !fir.ref<!fir.array<5x!fir.type<_QM__fortran_ieee_exceptionsTieee_flag_type{_QM__fortran_ieee_exceptionsTieee_flag_type.flag:i8}>>>
  ! CHECK:     %[[V_61:[0-9]+]] = fir.declare %[[V_60]](%[[V_1]]) {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro.5x_QM__fortran_ieee_exceptionsTieee_flag_type.0"} : (!fir.ref<!fir.array<5x!fir.type<_QM__fortran_ieee_exceptionsTieee_flag_type{_QM__fortran_ieee_exceptionsTieee_flag_type.flag:i8}>>>, !fir.shape<1>) -> !fir.ref<!fir.array<5x!fir.type<_QM__fortran_ieee_exceptionsTieee_flag_type{_QM__fortran_ieee_exceptionsTieee_flag_type.flag:i8}>>>
  ! CHECK:     fir.do_loop %arg0 = %c1{{.*}} to %c5{{.*}} step %c1{{.*}} {
  ! CHECK:       %[[V_95:[0-9]+]] = fir.array_coor %[[V_61]](%[[V_1]]) %arg0 : (!fir.ref<!fir.array<5x!fir.type<_QM__fortran_ieee_exceptionsTieee_flag_type{_QM__fortran_ieee_exceptionsTieee_flag_type.flag:i8}>>>, !fir.shape<1>, index) -> !fir.ref<!fir.type<_QM__fortran_ieee_exceptionsTieee_flag_type{_QM__fortran_ieee_exceptionsTieee_flag_type.flag:i8}>>
  ! CHECK:       %[[V_96:[0-9]+]] = fir.field_index _QM__fortran_ieee_exceptionsTieee_flag_type.flag, !fir.type<_QM__fortran_ieee_exceptionsTieee_flag_type{_QM__fortran_ieee_exceptionsTieee_flag_type.flag:i8}>
  ! CHECK:       %[[V_97:[0-9]+]] = fir.coordinate_of %[[V_95]], %[[V_96]] : (!fir.ref<!fir.type<_QM__fortran_ieee_exceptionsTieee_flag_type{_QM__fortran_ieee_exceptionsTieee_flag_type.flag:i8}>>, !fir.field) -> !fir.ref<i8>
  ! CHECK:       %[[V_98:[0-9]+]] = fir.load %[[V_97]] : !fir.ref<i8>
  ! CHECK:       %[[V_99:[0-9]+]] = fir.convert %[[V_98]] : (i8) -> i32
  ! CHECK:       %[[V_100:[0-9]+]] = fir.call @_FortranAMapException(%[[V_99]]) fastmath<contract> : (i32) -> i32
  ! CHECK:       fir.if %true{{[_0-9]*}} {
  ! CHECK:         %[[V_101:[0-9]+]] = fir.call @feenableexcept(%[[V_100]]) fastmath<contract> : (i32) -> i32
  ! CHECK:       } else {
  ! CHECK:         %[[V_101:[0-9]+]] = fir.call @fedisableexcept(%[[V_100]]) fastmath<contract> : (i32) -> i32
  ! CHECK:       }
  ! CHECK:     }
  call ieee_set_halting_mode(ieee_all, .true.)

  ! CHECK:     %[[V_62:[0-9]+]] = fir.declare %[[V_60]](%[[V_1]]) {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro.5x_QM__fortran_ieee_exceptionsTieee_flag_type.0"} : (!fir.ref<!fir.array<5x!fir.type<_QM__fortran_ieee_exceptionsTieee_flag_type{_QM__fortran_ieee_exceptionsTieee_flag_type.flag:i8}>>>, !fir.shape<1>) -> !fir.ref<!fir.array<5x!fir.type<_QM__fortran_ieee_exceptionsTieee_flag_type{_QM__fortran_ieee_exceptionsTieee_flag_type.flag:i8}>>>
  ! CHECK:     fir.do_loop %arg0 = %c1{{.*}} to %c5{{.*}} step %c1{{.*}} {
  ! CHECK:       %[[V_95:[0-9]+]] = fir.array_coor %[[V_62]](%[[V_1]]) %arg0 : (!fir.ref<!fir.array<5x!fir.type<_QM__fortran_ieee_exceptionsTieee_flag_type{_QM__fortran_ieee_exceptionsTieee_flag_type.flag:i8}>>>, !fir.shape<1>, index) -> !fir.ref<!fir.type<_QM__fortran_ieee_exceptionsTieee_flag_type{_QM__fortran_ieee_exceptionsTieee_flag_type.flag:i8}>>
  ! CHECK:       %[[V_96:[0-9]+]] = fir.array_coor %[[V_59]](%[[V_1]]) %arg0 : (!fir.ref<!fir.array<5x!fir.logical<4>>>, !fir.shape<1>, index) -> !fir.ref<!fir.logical<4>>
  ! CHECK:       %[[V_97:[0-9]+]] = fir.field_index _QM__fortran_ieee_exceptionsTieee_flag_type.flag, !fir.type<_QM__fortran_ieee_exceptionsTieee_flag_type{_QM__fortran_ieee_exceptionsTieee_flag_type.flag:i8}>
  ! CHECK:       %[[V_98:[0-9]+]] = fir.coordinate_of %[[V_95]], %[[V_97]] : (!fir.ref<!fir.type<_QM__fortran_ieee_exceptionsTieee_flag_type{_QM__fortran_ieee_exceptionsTieee_flag_type.flag:i8}>>, !fir.field) -> !fir.ref<i8>
  ! CHECK:       %[[V_99:[0-9]+]] = fir.load %[[V_98]] : !fir.ref<i8>
  ! CHECK:       %[[V_100:[0-9]+]] = fir.call @fegetexcept() fastmath<contract> : () -> i32
  ! CHECK:       %[[V_101:[0-9]+]] = fir.convert %[[V_99]] : (i8) -> i32
  ! CHECK:       %[[V_102:[0-9]+]] = fir.call @_FortranAMapException(%[[V_101]]) fastmath<contract> : (i32) -> i32
  ! CHECK:       %[[V_103:[0-9]+]] = arith.andi %[[V_100]], %[[V_102]] : i32
  ! CHECK:       %[[V_104:[0-9]+]] = arith.cmpi ne, %[[V_103]], %c0{{.*}} : i32
  ! CHECK:       %[[V_105:[0-9]+]] = fir.convert %[[V_104]] : (i1) -> !fir.logical<4>
  ! CHECK:       fir.store %[[V_105]] to %[[V_96]] : !fir.ref<!fir.logical<4>>
  ! CHECK:     }
  call ieee_get_halting_mode(ieee_all, v)

  print*, 'halting_mode [T T T T T] :', v

  ! CHECK:     %[[V_75:[0-9]+]] = fir.convert %[[V_57]] : (!fir.ref<!fir.type<_QM__fortran_ieee_exceptionsTieee_status_type{_QM__fortran_ieee_exceptionsTieee_status_type.__data:!fir.array<8xi32>}>>) -> !fir.ref<i32>
  ! CHECK:     %[[V_76:[0-9]+]] = fir.call @fegetenv(%[[V_75]]) fastmath<contract> : (!fir.ref<i32>) -> i32
  call ieee_get_status(status)

  ! CHECK:     %[[V_77:[0-9]+]] = fir.address_of(@_QQro.3x_QM__fortran_ieee_exceptionsTieee_flag_type.1) : !fir.ref<!fir.array<3x!fir.type<_QM__fortran_ieee_exceptionsTieee_flag_type{_QM__fortran_ieee_exceptionsTieee_flag_type.flag:i8}>>>
  ! CHECK:     %[[V_78:[0-9]+]] = fir.declare %[[V_77]](%[[V_54]]) {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro.3x_QM__fortran_ieee_exceptionsTieee_flag_type.1"} : (!fir.ref<!fir.array<3x!fir.type<_QM__fortran_ieee_exceptionsTieee_flag_type{_QM__fortran_ieee_exceptionsTieee_flag_type.flag:i8}>>>, !fir.shape<1>) -> !fir.ref<!fir.array<3x!fir.type<_QM__fortran_ieee_exceptionsTieee_flag_type{_QM__fortran_ieee_exceptionsTieee_flag_type.flag:i8}>>>
  ! CHECK:     fir.do_loop %arg0 = %c1{{.*}} to %c3{{.*}} step %c1{{.*}} {
  ! CHECK:       %[[V_95:[0-9]+]] = fir.array_coor %[[V_78]](%[[V_54]]) %arg0 : (!fir.ref<!fir.array<3x!fir.type<_QM__fortran_ieee_exceptionsTieee_flag_type{_QM__fortran_ieee_exceptionsTieee_flag_type.flag:i8}>>>, !fir.shape<1>, index) -> !fir.ref<!fir.type<_QM__fortran_ieee_exceptionsTieee_flag_type{_QM__fortran_ieee_exceptionsTieee_flag_type.flag:i8}>>
  ! CHECK:       %[[V_96:[0-9]+]] = fir.field_index _QM__fortran_ieee_exceptionsTieee_flag_type.flag, !fir.type<_QM__fortran_ieee_exceptionsTieee_flag_type{_QM__fortran_ieee_exceptionsTieee_flag_type.flag:i8}>
  ! CHECK:       %[[V_97:[0-9]+]] = fir.coordinate_of %[[V_95]], %[[V_96]] : (!fir.ref<!fir.type<_QM__fortran_ieee_exceptionsTieee_flag_type{_QM__fortran_ieee_exceptionsTieee_flag_type.flag:i8}>>, !fir.field) -> !fir.ref<i8>
  ! CHECK:       %[[V_98:[0-9]+]] = fir.load %[[V_97]] : !fir.ref<i8>
  ! CHECK:       %[[V_99:[0-9]+]] = fir.convert %[[V_98]] : (i8) -> i32
  ! CHECK:       %[[V_100:[0-9]+]] = fir.call @_FortranAMapException(%[[V_99]]) fastmath<contract> : (i32) -> i32
  ! CHECK:       fir.if %false{{[_0-9]*}} {
  ! CHECK:         %[[V_101:[0-9]+]] = fir.call @feenableexcept(%[[V_100]]) fastmath<contract> : (i32) -> i32
  ! CHECK:       } else {
  ! CHECK:         %[[V_101:[0-9]+]] = fir.call @fedisableexcept(%[[V_100]]) fastmath<contract> : (i32) -> i32
  ! CHECK:       }
  ! CHECK:     }
  call ieee_set_halting_mode(ieee_usual, .false.)

  ! CHECK:     %[[V_79:[0-9]+]] = fir.declare %[[V_60]](%[[V_1]]) {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro.5x_QM__fortran_ieee_exceptionsTieee_flag_type.0"} : (!fir.ref<!fir.array<5x!fir.type<_QM__fortran_ieee_exceptionsTieee_flag_type{_QM__fortran_ieee_exceptionsTieee_flag_type.flag:i8}>>>, !fir.shape<1>) -> !fir.ref<!fir.array<5x!fir.type<_QM__fortran_ieee_exceptionsTieee_flag_type{_QM__fortran_ieee_exceptionsTieee_flag_type.flag:i8}>>>
  ! CHECK:     fir.do_loop %arg0 = %c1{{.*}} to %c5{{.*}} step %c1{{.*}} {
  ! CHECK:       %[[V_95:[0-9]+]] = fir.array_coor %[[V_79]](%[[V_1]]) %arg0 : (!fir.ref<!fir.array<5x!fir.type<_QM__fortran_ieee_exceptionsTieee_flag_type{_QM__fortran_ieee_exceptionsTieee_flag_type.flag:i8}>>>, !fir.shape<1>, index) -> !fir.ref<!fir.type<_QM__fortran_ieee_exceptionsTieee_flag_type{_QM__fortran_ieee_exceptionsTieee_flag_type.flag:i8}>>
  ! CHECK:       %[[V_96:[0-9]+]] = fir.array_coor %[[V_59]](%[[V_1]]) %arg0 : (!fir.ref<!fir.array<5x!fir.logical<4>>>, !fir.shape<1>, index) -> !fir.ref<!fir.logical<4>>
  ! CHECK:       %[[V_97:[0-9]+]] = fir.field_index _QM__fortran_ieee_exceptionsTieee_flag_type.flag, !fir.type<_QM__fortran_ieee_exceptionsTieee_flag_type{_QM__fortran_ieee_exceptionsTieee_flag_type.flag:i8}>
  ! CHECK:       %[[V_98:[0-9]+]] = fir.coordinate_of %[[V_95]], %[[V_97]] : (!fir.ref<!fir.type<_QM__fortran_ieee_exceptionsTieee_flag_type{_QM__fortran_ieee_exceptionsTieee_flag_type.flag:i8}>>, !fir.field) -> !fir.ref<i8>
  ! CHECK:       %[[V_99:[0-9]+]] = fir.load %[[V_98]] : !fir.ref<i8>
  ! CHECK:       %[[V_100:[0-9]+]] = fir.call @fegetexcept() fastmath<contract> : () -> i32
  ! CHECK:       %[[V_101:[0-9]+]] = fir.convert %[[V_99]] : (i8) -> i32
  ! CHECK:       %[[V_102:[0-9]+]] = fir.call @_FortranAMapException(%[[V_101]]) fastmath<contract> : (i32) -> i32
  ! CHECK:       %[[V_103:[0-9]+]] = arith.andi %[[V_100]], %[[V_102]] : i32
  ! CHECK:       %[[V_104:[0-9]+]] = arith.cmpi ne, %[[V_103]], %c0{{.*}} : i32
  ! CHECK:       %[[V_105:[0-9]+]] = fir.convert %[[V_104]] : (i1) -> !fir.logical<4>
  ! CHECK:       fir.store %[[V_105]] to %[[V_96]] : !fir.ref<!fir.logical<4>>
  ! CHECK:     }
  call ieee_get_halting_mode(ieee_all, v)

  print*, 'halting_mode [F F F T T] :', v

  ! CHECK:     %[[V_87:[0-9]+]] = fir.call @fesetenv(%[[V_75]]) fastmath<contract> : (!fir.ref<i32>) -> i32
  ! CHECK:     %[[V_88:[0-9]+]] = fir.declare %[[V_60]](%[[V_1]]) {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro.5x_QM__fortran_ieee_exceptionsTieee_flag_type.0"} : (!fir.ref<!fir.array<5x!fir.type<_QM__fortran_ieee_exceptionsTieee_flag_type{_QM__fortran_ieee_exceptionsTieee_flag_type.flag:i8}>>>, !fir.shape<1>) -> !fir.ref<!fir.array<5x!fir.type<_QM__fortran_ieee_exceptionsTieee_flag_type{_QM__fortran_ieee_exceptionsTieee_flag_type.flag:i8}>>>
  call ieee_set_status(status)

  ! CHECK:     fir.do_loop %arg0 = %c1{{.*}} to %c5{{.*}} step %c1{{.*}} {
  ! CHECK:       %[[V_95:[0-9]+]] = fir.array_coor %[[V_88]](%[[V_1]]) %arg0 : (!fir.ref<!fir.array<5x!fir.type<_QM__fortran_ieee_exceptionsTieee_flag_type{_QM__fortran_ieee_exceptionsTieee_flag_type.flag:i8}>>>, !fir.shape<1>, index) -> !fir.ref<!fir.type<_QM__fortran_ieee_exceptionsTieee_flag_type{_QM__fortran_ieee_exceptionsTieee_flag_type.flag:i8}>>
  ! CHECK:       %[[V_96:[0-9]+]] = fir.array_coor %[[V_59]](%[[V_1]]) %arg0 : (!fir.ref<!fir.array<5x!fir.logical<4>>>, !fir.shape<1>, index) -> !fir.ref<!fir.logical<4>>
  ! CHECK:       %[[V_97:[0-9]+]] = fir.field_index _QM__fortran_ieee_exceptionsTieee_flag_type.flag, !fir.type<_QM__fortran_ieee_exceptionsTieee_flag_type{_QM__fortran_ieee_exceptionsTieee_flag_type.flag:i8}>
  ! CHECK:       %[[V_98:[0-9]+]] = fir.coordinate_of %[[V_95]], %[[V_97]] : (!fir.ref<!fir.type<_QM__fortran_ieee_exceptionsTieee_flag_type{_QM__fortran_ieee_exceptionsTieee_flag_type.flag:i8}>>, !fir.field) -> !fir.ref<i8>
  ! CHECK:       %[[V_99:[0-9]+]] = fir.load %[[V_98]] : !fir.ref<i8>
  ! CHECK:       %[[V_100:[0-9]+]] = fir.call @fegetexcept() fastmath<contract> : () -> i32
  ! CHECK:       %[[V_101:[0-9]+]] = fir.convert %[[V_99]] : (i8) -> i32
  ! CHECK:       %[[V_102:[0-9]+]] = fir.call @_FortranAMapException(%[[V_101]]) fastmath<contract> : (i32) -> i32
  ! CHECK:       %[[V_103:[0-9]+]] = arith.andi %[[V_100]], %[[V_102]] : i32
  ! CHECK:       %[[V_104:[0-9]+]] = arith.cmpi ne, %[[V_103]], %c0{{.*}} : i32
  ! CHECK:       %[[V_105:[0-9]+]] = fir.convert %[[V_104]] : (i1) -> !fir.logical<4>
  ! CHECK:       fir.store %[[V_105]] to %[[V_96]] : !fir.ref<!fir.logical<4>>
  ! CHECK:     }
  call ieee_get_halting_mode(ieee_all, v)

  print*, 'halting_mode [T T T T T] :', v
end
