! Test for PassBy::BaseAddressValueAttribute
! RUN: bbc -emit-hlfir %s -o - | FileCheck %s

!CHECK-LABEL: func @_QQmain()
program call_by_value_attr
  interface
     subroutine subri(val)
       integer, value :: val
     end subroutine subri
     subroutine subra(val)
       integer, value, dimension(10) :: val
     end subroutine subra
  end interface

  integer :: v
  integer, dimension(10) :: a
  integer, dimension(15) :: b
  ! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare{{.*}}a"
  ! CHECK:           %[[VAL_7:.*]]:2 = hlfir.declare{{.*}}b"
  ! CHECK:           %[[VAL_9:.*]]:2 = hlfir.declare{{.*}}v"
  v = 17
  call subri(v)
  ! CHECK:           %[[VAL_11:.*]] = fir.load %[[VAL_9]]#0 : !fir.ref<i32>
  ! CHECK:           fir.call @_QPsubri(%[[VAL_11]]) fastmath<contract> : (i32) -> ()
  a = (/ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 /)
  call subra(a)
  ! CHECK:           %[[VAL_16:.*]] = hlfir.as_expr %[[VAL_3]]#0 : (!fir.ref<!fir.array<10xi32>>) -> !hlfir.expr<10xi32>
  ! CHECK:           %[[VAL_17:.*]]:3 = hlfir.associate %[[VAL_16]](%{{.*}}) {adapt.valuebyref} : (!hlfir.expr<10xi32>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>, i1)
  ! CHECK:           fir.call @_QPsubra(%[[VAL_17]]#0) fastmath<contract> : (!fir.ref<!fir.array<10xi32>>) -> ()

  b = (/ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 /)
  call subra(b(5:14))
  ! CHECK:           %[[VAL_27:.*]] = hlfir.designate %[[VAL_7]]#0
  ! CHECK:           %[[VAL_28:.*]] = hlfir.as_expr %[[VAL_27]] : (!fir.ref<!fir.array<10xi32>>) -> !hlfir.expr<10xi32>
  ! CHECK:           %[[VAL_29:.*]]:3 = hlfir.associate %[[VAL_28]](%{{.*}}) {adapt.valuebyref} : (!hlfir.expr<10xi32>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>, i1)
  ! CHECK:           fir.call @_QPsubra(%[[VAL_29]]#0) fastmath<contract> : (!fir.ref<!fir.array<10xi32>>) -> ()
end program call_by_value_attr

subroutine subri(val)
  integer, value :: val
  call test_numeric_scalar_value(val)
end subroutine subri
! CHECK-LABEL:   func.func @_QPsubri(
! CHECK-SAME:                        %[[VAL_0:.*]]: i32 {fir.bindc_name = "val"}) {
! CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_2:.*]] = fir.alloca i32
! CHECK:           fir.store %[[VAL_0]] to %[[VAL_2]] : !fir.ref<i32>
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]] dummy_scope %[[VAL_1]] arg {{[0-9]+}} {fortran_attrs = #fir.var_attrs<value>, uniq_name = "_QFsubriEval"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           fir.call @_QPtest_numeric_scalar_value(%[[VAL_3]]#0) fastmath<contract> : (!fir.ref<i32>) -> ()
! CHECK:           return
! CHECK:         }

! CHECK-LABEL: func @_QPtest_litteral_copies_1
subroutine test_litteral_copies_1
  ! VALUE arguments  can be modified by the callee, so the static storage of
  ! literal constants and named parameters must not be passed directly to them.
  interface
    subroutine takes_array_value(v)
      integer, value :: v(4)
    end subroutine
  end interface
  integer, parameter :: p(100) = 42
  call takes_array_value(p)
  ! CHECK:           %[[VAL_7:.*]]:2 = hlfir.declare {{.*}}"_QQro.100xi4.2"
  ! CHECK:           %[[VAL_8:.*]] = hlfir.as_expr %[[VAL_7]]#0 : (!fir.ref<!fir.array<100xi32>>) -> !hlfir.expr<100xi32>
  ! CHECK:           %[[VAL_9:.*]]:3 = hlfir.associate %[[VAL_8]](%{{.*}}) {adapt.valuebyref} : (!hlfir.expr<100xi32>, !fir.shape<1>) -> (!fir.ref<!fir.array<100xi32>>, !fir.ref<!fir.array<100xi32>>, i1)
  ! CHECK:           %[[VAL_10:.*]] = fir.convert %[[VAL_9]]#0 : (!fir.ref<!fir.array<100xi32>>) -> !fir.ref<!fir.array<4xi32>>
  ! CHECK:           fir.call @_QPtakes_array_value(%[[VAL_10]]) fastmath<contract> : (!fir.ref<!fir.array<4xi32>>) -> ()
end subroutine

! CHECK-LABEL: func @_QPtest_litteral_copies_2
subroutine test_litteral_copies_2
  interface
    subroutine takes_char_value(v)
      character(*), value :: v
    end subroutine
  end interface
  call takes_char_value("a character string litteral that could be locally modfied by the callee")
  ! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare {{.*}}"_QQcl
  ! CHECK:           %[[VAL_3:.*]] = hlfir.as_expr %[[VAL_2]]#0 : (!fir.ref<!fir.char<1,71>>) -> !hlfir.expr<!fir.char<1,71>>
  ! CHECK:           %[[VAL_4:.*]]:3 = hlfir.associate %[[VAL_3]] typeparams %{{.*}} {adapt.valuebyref} : (!hlfir.expr<!fir.char<1,71>>, index) -> (!fir.ref<!fir.char<1,71>>, !fir.ref<!fir.char<1,71>>, i1)
  ! CHECK:           %[[VAL_5:.*]] = fir.emboxchar %[[VAL_4]]#0, %{{.*}} : (!fir.ref<!fir.char<1,71>>, index) -> !fir.boxchar<1>
  ! CHECK:           fir.call @_QPtakes_char_value(%[[VAL_5]]) fastmath<contract> : (!fir.boxchar<1>) -> ()
end subroutine
