! Test ignore_tkr(tk) with character dummies
! RUN: %flang_fc1 -emit-fir %s -o - | FileCheck %s

module test_char_tk
  interface
    subroutine foo(c)
    character(1)::c(*)
    !dir$ ignore_tkr(tkrdm) c
    end subroutine
  end interface
  interface
    subroutine foo_requires_explicit_interface(c, i)
    character(1)::c(*)
    !dir$ ignore_tkr(tkrdm) c
    integer, optional :: i
    end subroutine
  end interface
contains
  subroutine test_normal()
    character(1) :: c(10)
    call foo(c)
  end subroutine
!CHECK-LABEL:   func.func @_QMtest_char_tkPtest_normal(
!CHECK:           %[[VAL_6:.*]] = fir.emboxchar %{{.*}}, %c1{{.*}}: (!fir.ref<!fir.char<1>>, index) -> !fir.boxchar<1>
!CHECK:           fir.call @_QPfoo(%[[VAL_6]]) fastmath<contract> : (!fir.boxchar<1>) -> ()
  subroutine test_normal2()
    character(10) :: c(10)
    call foo(c)
  end subroutine
!CHECK-LABEL:   func.func @_QMtest_char_tkPtest_normal2(
!CHECK:           %[[VAL_4:.*]] = fir.convert %{{.*}} : (!fir.ref<!fir.array<10x!fir.char<1,10>>>) -> !fir.ref<!fir.char<1,10>>
!CHECK:           %[[VAL_5:.*]] = fir.emboxchar %[[VAL_4]], %c10{{.*}}: (!fir.ref<!fir.char<1,10>>, index) -> !fir.boxchar<1>
!CHECK:           fir.call @_QPfoo(%[[VAL_5]]) fastmath<contract> : (!fir.boxchar<1>) -> ()
  subroutine test_weird()
    real :: c(10)
    call foo(c)
  end subroutine
!CHECK-LABEL:   func.func @_QMtest_char_tkPtest_weird(
!CHECK:           %[[VAL_5:.*]] = fir.convert %{{.*}} : (!fir.ref<!fir.array<10xf32>>) -> !fir.ref<!fir.char<1,?>>
!CHECK:           %[[VAL_6:.*]] = fir.emboxchar %[[VAL_5]], %c0{{.*}}: (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
!CHECK:           fir.call @_QPfoo(%[[VAL_6]]) fastmath<contract> : (!fir.boxchar<1>) -> ()

  subroutine test_requires_explicit_interface(x, i)
    real :: x(10)
    integer :: i
    call foo_requires_explicit_interface(x, i)
  end subroutine
!CHECK-LABEL:   func.func @_QMtest_char_tkPtest_requires_explicit_interface(
!CHECK:           %[[VAL_5:.*]] = fir.convert %{{.*}} : (!fir.ref<!fir.array<10xf32>>) -> !fir.ref<!fir.char<1,?>>
!CHECK:           %[[VAL_6:.*]] = fir.emboxchar %[[VAL_5]], %c0{{.*}}: (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
!CHECK:           fir.call @_QPfoo_requires_explicit_interface(%[[VAL_6]], %{{.*}})
end module
