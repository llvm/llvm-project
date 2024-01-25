! RUN: bbc -emit-fir -o - %s | FileCheck %s

! CHECK-LABEL: c.func @_QPout
subroutine out(x)
  use ieee_arithmetic
  integer, parameter :: k = 8

  ! CHECK:     %[[V_60:[0-9]+]] = fir.alloca !fir.logical<4> {bindc_name = "l", uniq_name = "_QFoutEl"}
  ! CHECK:     %[[V_61:[0-9]+]] = fir.declare %[[V_60]] {uniq_name = "_QFoutEl"} : (!fir.ref<!fir.logical<4>>) -> !fir.ref<!fir.logical<4>>
  ! CHECK:     %[[V_62:[0-9]+]] = fir.alloca f64 {bindc_name = "r", uniq_name = "_QFoutEr"}
  ! CHECK:     %[[V_63:[0-9]+]] = fir.declare %[[V_62]] {uniq_name = "_QFoutEr"} : (!fir.ref<f64>) -> !fir.ref<f64>
  ! CHECK:     %[[V_64:[0-9]+]] = fir.declare %arg0 {uniq_name = "_QFoutEx"} : (!fir.ref<f64>) -> !fir.ref<f64>
  real(k) :: x, r
  logical :: L

  ! CHECK:     %[[V_65:[0-9]+]] = fir.address_of(@_QQro._QM__fortran_ieee_exceptionsTieee_flag_type.0) : !fir.ref<!fir.type<_QM__fortran_ieee_exceptionsTieee_flag_type{_QM__fortran_ieee_exceptionsTieee_flag_type.flag:i8}>>
  ! CHECK:     %[[V_66:[0-9]+]] = fir.declare %[[V_65]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro._QM__fortran_ieee_exceptionsTieee_flag_type.0"} : (!fir.ref<!fir.type<_QM__fortran_ieee_exceptionsTieee_flag_type{_QM__fortran_ieee_exceptionsTieee_flag_type.flag:i8}>>) -> !fir.ref<!fir.type<_QM__fortran_ieee_exceptionsTieee_flag_type{_QM__fortran_ieee_exceptionsTieee_flag_type.flag:i8}>>
  ! CHECK:     %[[V_67:[0-9]+]] = fir.field_index _QM__fortran_ieee_exceptionsTieee_flag_type.flag, !fir.type<_QM__fortran_ieee_exceptionsTieee_flag_type{_QM__fortran_ieee_exceptionsTieee_flag_type.flag:i8}>
  ! CHECK:     %[[V_68:[0-9]+]] = fir.coordinate_of %[[V_66]], %[[V_67]] : (!fir.ref<!fir.type<_QM__fortran_ieee_exceptionsTieee_flag_type{_QM__fortran_ieee_exceptionsTieee_flag_type.flag:i8}>>, !fir.field) -> !fir.ref<i8>
  ! CHECK:     %[[V_69:[0-9]+]] = fir.load %[[V_68]] : !fir.ref<i8>
  ! CHECK:     %[[V_70:[0-9]+]] = fir.convert %[[V_69]] : (i8) -> i32
  ! CHECK:     %[[V_71:[0-9]+]] = fir.call @_FortranAMapException(%[[V_70]]) fastmath<contract> : (i32) -> i32
  ! CHECK:     fir.if %false{{[_0-9]*}} {
  ! CHECK:       %[[V_101:[0-9]+]] = fir.call @feraiseexcept(%[[V_71]]) fastmath<contract> : (i32) -> i32
  ! CHECK:     } else {
  ! CHECK:       %[[V_101:[0-9]+]] = fir.call @feclearexcept(%[[V_71]]) fastmath<contract> : (i32) -> i32
  ! CHECK:     }
  call ieee_set_flag(ieee_divide_by_zero, .false.)

  ! CHECK:     %[[V_72:[0-9]+]] = fir.load %[[V_64]] : !fir.ref<f64>
  ! CHECK:     %[[V_73:[0-9]+]] = arith.bitcast %[[V_72]] : f64 to i64
  ! CHECK:     %[[V_74:[0-9]+]] = arith.cmpf oeq, %[[V_72]], %cst{{[_0-9]*}} {{.*}} : f64
  ! CHECK:     %[[V_75:[0-9]+]] = fir.if %[[V_74]] -> (f64) {
  ! CHECK:       %[[V_101:[0-9]+]] = fir.call @_FortranAMapException(%c4{{.*}}) fastmath<contract> : (i32) -> i32
  ! CHECK:       %[[V_102:[0-9]+]] = fir.call @feraiseexcept(%[[V_101]]) fastmath<contract> : (i32) -> i32
  ! CHECK:       fir.result %cst{{[_0-9]*}} : f64
  ! CHECK:     } else {
  ! CHECK:       %[[V_101:[0-9]+]] = arith.shli %[[V_73]], %c1{{.*}} : i64
  ! CHECK:       %[[V_102:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_72]]) <{bit = 504 : i32}> : (f64) -> i1
  ! CHECK:       %[[V_103:[0-9]+]] = fir.if %[[V_102]] -> (f64) {
  ! CHECK:         %[[V_104:[0-9]+]] = arith.shrui %[[V_101]], %c53{{.*}} : i64
  ! CHECK:         %[[V_105:[0-9]+]] = arith.subi %[[V_104]], %c1023{{.*}} : i64
  ! CHECK:         %[[V_106:[0-9]+]] = fir.convert %[[V_105]] : (i64) -> f64
  ! CHECK:         fir.result %[[V_106]] : f64
  ! CHECK:       } else {
  ! CHECK:         %[[V_104:[0-9]+]] = arith.shrui %[[V_101]], %c1{{.*}} : i64
  ! CHECK:         %[[V_105:[0-9]+]] = arith.bitcast %[[V_104]] : i64 to f64
  ! CHECK:         fir.result %[[V_105]] : f64
  ! CHECK:       }
  ! CHECK:       fir.result %[[V_103]] : f64
  ! CHECK:     }
  ! CHECK:     fir.store %[[V_75]] to %[[V_63]] : !fir.ref<f64>
  r = ieee_logb(x)

  ! CHECK:     %[[V_76:[0-9]+]] = fir.declare %[[V_65]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro._QM__fortran_ieee_exceptionsTieee_flag_type.0"} : (!fir.ref<!fir.type<_QM__fortran_ieee_exceptionsTieee_flag_type{_QM__fortran_ieee_exceptionsTieee_flag_type.flag:i8}>>) -> !fir.ref<!fir.type<_QM__fortran_ieee_exceptionsTieee_flag_type{_QM__fortran_ieee_exceptionsTieee_flag_type.flag:i8}>>
  ! CHECK:     %[[V_77:[0-9]+]] = fir.coordinate_of %[[V_76]], %[[V_67]] : (!fir.ref<!fir.type<_QM__fortran_ieee_exceptionsTieee_flag_type{_QM__fortran_ieee_exceptionsTieee_flag_type.flag:i8}>>, !fir.field) -> !fir.ref<i8>
  ! CHECK:     %[[V_78:[0-9]+]] = fir.load %[[V_77]] : !fir.ref<i8>
  ! CHECK:     %[[V_79:[0-9]+]] = fir.convert %[[V_78]] : (i8) -> i32
  ! CHECK:     %[[V_80:[0-9]+]] = fir.call @_FortranAMapException(%[[V_79]]) fastmath<contract> : (i32) -> i32
  ! CHECK:     %[[V_81:[0-9]+]] = fir.call @fetestexcept(%[[V_80]]) fastmath<contract> : (i32) -> i32
  ! CHECK:     %[[V_82:[0-9]+]] = arith.cmpi ne, %[[V_81]], %c0{{.*}} : i32
  ! CHECK:     %[[V_83:[0-9]+]] = fir.convert %[[V_82]] : (i1) -> !fir.logical<4>
  ! CHECK:     fir.store %[[V_83]] to %[[V_61]] : !fir.ref<!fir.logical<4>>
  call ieee_get_flag(ieee_divide_by_zero, L)

  8 format ('  kind=8  ', f12.2, z18.16, f9.1, l3, '  ')
  write(*, 8) x, x, r, L
end

  use ieee_arithmetic
  integer, parameter :: k = 8
  real(k) :: x, r

  call out(ieee_value(x, ieee_signaling_nan))
  call out(ieee_value(x, ieee_quiet_nan))
  call out(ieee_value(x, ieee_negative_inf))
  call out( -huge(x))
  call out( -huge(x)/2)
  call out(-sqrt(huge(x)))
  call out(-2000.0_k)
  call out(   -9.9_k)
  call out(   -9.0_k)
  call out(   -8.0_k)
  call out(   -7.0_k)
  call out(   -6.0_k)
  call out(   -5.0_k)
  call out(   -4.0_k)
  call out(   -3.9_k)
  call out(   -3.0_k)
  call out(   -2.0_k)
  call out(   -1.1_k)
  call out(ieee_value(x, ieee_negative_normal))
  call out( -.0001_k)
  call out( -tiny(x))
  call out(ieee_value(x, ieee_negative_subnormal))
  call out(ieee_value(x, ieee_negative_zero))
  call out(ieee_value(x, ieee_positive_zero))
  call out(ieee_value(x, ieee_positive_subnormal))
  call out(tiny(x))
  call out(.0001_k)
  call out(ieee_value(x, ieee_positive_normal))
  call out(   1.1_k)
  call out(   2.0_k)
  call out(   3.0_k)
  call out(   3.9_k)
  call out(   4.0_k)
  call out(   5.0_k)
  call out(   6.0_k)
  call out(   7.0_k)
  call out(   8.0_k)
  call out(   9.0_k)
  call out(   9.9_k)
  call out(2000.0_k)
  call out( sqrt(huge(x)))
  call out( huge(x)/2)
  call out( huge(x))
  call out(ieee_value(x, ieee_positive_inf))
end
