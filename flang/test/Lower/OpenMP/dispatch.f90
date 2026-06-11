!RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s --check-prefix=HLFIR
!RUN: %flang_fc1 -emit-mlir -fopenmp %s -o - | FileCheck %s --check-prefix=FIR

!HLFIR-LABEL: func @_QMfuncsPfoo_dispatch
!HLFIR: %[[XD_H:.*]]:2 = hlfir.declare %{{.*}} {{{.*}}uniq_name = {{.*}}foo_dispatch{{.*}}x{{.*}}
!HLFIR: %[[LOAD_H:.*]] = fir.load %[[XD_H]]#0 : !fir.ref<i32>
!HLFIR: %[[C1_H:.*]] = arith.constant 1 : i32
!HLFIR: %[[CMP_H:.*]] = arith.cmpi eq, %[[LOAD_H]], %[[C1_H]] : i32
!HLFIR: fir.if %[[CMP_H]] {
!HLFIR:   fir.call @_QMfuncsPvariant1() {{.*}}: () -> ()
!HLFIR: } else {
!HLFIR:   fir.call @_QMfuncsPvariant2() {{.*}}: () -> ()
!HLFIR: }

!FIR-LABEL: func @_QMfuncsPfoo_dispatch
!FIR: %[[XD_F:.*]] = fir.declare %{{.*}} {{{.*}}uniq_name = {{.*}}foo_dispatch{{.*}}x{{.*}}
!FIR: %[[LOAD_F:.*]] = fir.load %[[XD_F]] : !fir.ref<i32>
!FIR: %[[CMP_F:.*]] = arith.cmpi eq, %[[LOAD_F]], %{{.*}} : i32
!FIR: fir.if %[[CMP_F]] {
!FIR:   fir.call @_QMfuncsPvariant1() {{.*}}: () -> ()
!FIR: } else {
!FIR:   fir.call @_QMfuncsPvariant2() {{.*}}: () -> ()
!FIR: }

module funcs
  implicit none

contains

  subroutine variant1()
    print *, "in variant1"
  end subroutine

  subroutine variant2()
    print *, "in variant2"
  end subroutine

  subroutine foo_dispatch(x)
    integer, intent(in) :: x
    if (x == 1) then
      call variant1()
    else
      call variant2()
    end if
  end subroutine

end module funcs

!HLFIR-LABEL: func @_QQmain
!FIR-LABEL: func @_QQmain
!FIR-DAG: %[[C1:.*]] = arith.constant 1 : i32
!FIR-DAG: %[[C2:.*]] = arith.constant 2 : i32
!FIR: %[[X:.*]] = fir.declare %{{.*}} {uniq_name = {{.*}}x{{.*}}
program dispatch_test
  use funcs
  implicit none
  integer :: x

  !HLFIR: %[[X:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = {{.*}}x{{.*}}
  !HLFIR: %[[C1:.*]] = arith.constant 1 : i32
  !HLFIR: hlfir.assign %[[C1]] to %[[X]]#0 : i32, !fir.ref<i32>
  !FIR: fir.store %[[C1]] to %[[X]] : !fir.ref<i32>
  x = 1
  !HLFIR: omp.dispatch {
  !FIR: omp.dispatch {
  !$omp dispatch
  !HLFIR:   fir.call @_QMfuncsPfoo_dispatch(%[[X]]#0) {{.*}}: (!fir.ref<i32>) -> ()
  !FIR:   fir.call @_QMfuncsPfoo_dispatch(%[[X]]) {{.*}}: (!fir.ref<i32>) -> ()
    call foo_dispatch(x)
  !HLFIR:   omp.terminator
  !FIR:   omp.terminator
  !HLFIR: }
  !FIR: }

  !HLFIR: %[[C2:.*]] = arith.constant 2 : i32
  !HLFIR: hlfir.assign %[[C2]] to %[[X]]#0 : i32, !fir.ref<i32>
  !FIR: fir.store %[[C2]] to %[[X]] : !fir.ref<i32>
  x = 2
  !HLFIR: omp.dispatch {
  !FIR: omp.dispatch {
  !$omp dispatch
  !HLFIR:   fir.call @_QMfuncsPfoo_dispatch(%[[X]]#0) {{.*}}: (!fir.ref<i32>) -> ()
  !FIR:   fir.call @_QMfuncsPfoo_dispatch(%[[X]]) {{.*}}: (!fir.ref<i32>) -> ()
    call foo_dispatch(x)
  !HLFIR:   omp.terminator
  !FIR:   omp.terminator
  !HLFIR: }
  !FIR: }
end program
