! Test for PassBy::Value
! RUN: bbc -emit-fir %s -o - | FileCheck %s

!CHECK-LABEL: func @_QQmain()
!CHECK: %false = arith.constant false
!CHECK: %[[LOGICAL_ALLOC:.*]] = fir.alloca !fir.logical<1>
!CHECK: %[[LOGICAL:.*]] = fir.declare %[[LOGICAL_ALLOC]]
!CHECK: %[[VALUE:.*]] = fir.convert %false : (i1) -> !fir.logical<1>
!CHECK: fir.store %[[VALUE]] to %[[LOGICAL]]
!CHECK: %[[LOAD:.*]] = fir.load %[[LOGICAL]]
!CHECK: fir.call @omp_set_nested(%[[LOAD]]) {{.*}}: {{.*}}

program call_by_value
  use iso_c_binding, only: c_bool
  interface
     subroutine omp_set_nested(enable) bind(c)
       import c_bool
       logical(c_bool), value :: enable
     end subroutine omp_set_nested
  end interface

  logical(c_bool) do_nested
  do_nested = .FALSE.
  call omp_set_nested(do_nested)
end program call_by_value

! CHECK-LABEL:   func.func @test_integer_value(
! CHECK-SAME:                                  %[[VAL_0:.*]]: i32
! CHECK:           %[[VAL_1:.*]] = fir.alloca i32
! CHECK:           fir.store %[[VAL_0]] to %[[VAL_1]] : !fir.ref<i32>
! CHECK:           %[[VAL_2:.*]] = fir.declare %[[VAL_1]]
! CHECK:           fir.call @_QPinternal_call(%[[VAL_2]]) {{.*}}: (!fir.ref<i32>) -> ()
! CHECK:           return
! CHECK:         }

subroutine test_integer_value(x) bind(c)
  integer, value :: x
  call internal_call(x)
end
! CHECK-LABEL:   func.func @test_real_value(
! CHECK-SAME:                               %[[VAL_0:.*]]: f32
! CHECK:           %[[VAL_1:.*]] = fir.alloca f32
! CHECK:           fir.store %[[VAL_0]] to %[[VAL_1]] : !fir.ref<f32>
! CHECK:           %[[VAL_2:.*]] = fir.declare %[[VAL_1]]
! CHECK:           fir.call @_QPinternal_call2(%[[VAL_2]]) {{.*}}: (!fir.ref<f32>) -> ()
! CHECK:           return
! CHECK:         }


subroutine test_real_value(x) bind(c)
  real, value :: x
  call internal_call2(x)
end
! CHECK-LABEL:   func.func @test_complex_value(
! CHECK-SAME:                                  %[[VAL_0:.*]]: complex<f32>
! CHECK:           %[[VAL_1:.*]] = fir.alloca complex<f32>
! CHECK:           fir.store %[[VAL_0]] to %[[VAL_1]] : !fir.ref<complex<f32>>
! CHECK:           %[[VAL_2:.*]] = fir.declare %[[VAL_1]]
! CHECK:           fir.call @_QPinternal_call3(%[[VAL_2]]) {{.*}}: (!fir.ref<complex<f32>>) -> ()
! CHECK:           return
! CHECK:         }


subroutine test_complex_value(x) bind(c)
  complex, value :: x
  call internal_call3(x)
end

! CHECK-LABEL:   func.func @test_char_value(
! CHECK-SAME:                               %[[VAL_0:.*]]: !fir.char<1>
! CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_2:.*]] = fir.alloca !fir.char<1>
! CHECK:           fir.store %[[VAL_0]] to %[[VAL_2]] : !fir.ref<!fir.char<1>>
! CHECK:           %[[VAL_3:.*]] = fir.declare %[[VAL_2]] typeparams %[[VAL_1]]
! CHECK:           %[[VAL_4:.*]] = fir.emboxchar %[[VAL_3]], %[[VAL_1]] : (!fir.ref<!fir.char<1>>, index) -> !fir.boxchar<1>
! CHECK:           fir.call @_QPinternal_call4(%[[VAL_4]]) {{.*}}: (!fir.boxchar<1>) -> ()
! CHECK:           return
! CHECK:         }

subroutine test_char_value(x) bind(c)
  character(1), value :: x
  call internal_call4(x)
end

! CHECK-LABEL:   func.func @_QPtest_call_char_value(
! CHECK-SAME:                                       %[[VAL_0:.*]]: !fir.boxchar<1>
! CHECK:           %[[VAL_1:.*]]:2 = fir.unboxchar %[[VAL_0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:           %[[VAL_2:.*]] = fir.declare %[[VAL_1]]#0 typeparams %[[VAL_1]]#1
! CHECK:           %[[VAL_3:.*]] = fir.emboxchar %[[VAL_2]], %[[VAL_1]]#1 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
! CHECK:           %[[VAL_4:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1>>
! CHECK:           %[[VAL_5:.*]] = fir.load %[[VAL_4]] : !fir.ref<!fir.char<1>>
! CHECK:           fir.call @test_char_value(%[[VAL_5]]) {{.*}}: (!fir.char<1>) -> ()
! CHECK:           return
! CHECK:         }
subroutine test_call_char_value(x)
  character(*) :: x
  interface
    subroutine test_char_value(x) bind(c)
      character(1), value :: x
    end
  end interface
  call test_char_value(x)
end subroutine

! CHECK-LABEL:   func.func @_QPtest_cptr_value(
! CHECK-SAME:                                  %[[VAL_0:.*]]: !fir.ref<i64>
! CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK:           %[[VAL_2:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK:           %[[VAL_3:.*]] = fir.coordinate_of %[[VAL_1]], %[[VAL_2]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.field) -> !fir.ref<i64>
! CHECK:           %[[VAL_4:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<i64>) -> i64
! CHECK:           fir.store %[[VAL_4]] to %[[VAL_3]] : !fir.ref<i64>
! CHECK:           %[[VAL_5:.*]] = fir.declare %[[VAL_1]]
! CHECK:           fir.call @_QPinternal_call5(%[[VAL_5]]) fastmath<contract> : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>) -> ()
! CHECK:           return
! CHECK:         }

subroutine test_cptr_value(x)
  use iso_c_binding, only: c_ptr
  type(c_ptr), value :: x
  call internal_call5(x)
end
