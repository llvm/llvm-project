!RUN: bbc -emit-hlfir -fopenacc -fopenmp -o - %s | FileCheck %s

!https://github.com/llvm/llvm-project/issues/91526

! The if-construct is unstructured (stop in else) but wrappable, so the
! PFT-to-MLIR pass hides its CFG inside scf.execute_region. The THEN branch
! correctly branches to the wrap's merge (yield) block rather than falling
! into the ELSE block — the original bug fixed by issue 91526 stays fixed.
!CHECK-LABEL: func.func @_QPsimple1
!CHECK:   scf.execute_region
!CHECK:   cf.cond_br %{{[0-9]+}}, ^bb[[THEN:[0-9]+]], ^bb[[ELSE:[0-9]+]]
!CHECK: ^bb[[THEN]]:
!CHECK:   omp.parallel
!CHECK:   cf.br ^bb[[MERGE:[0-9]+]]
!CHECK: ^bb[[ELSE]]:
!CHECK:   fir.call @_FortranAStopStatement
!CHECK:   fir.unreachable
!CHECK: ^bb[[MERGE]]:
!CHECK:   scf.yield
!CHECK:   return

subroutine simple1(y)
  implicit none
  logical, intent(in) :: y
  integer :: i
  if (y) then
    !$omp parallel
    i = 1
    !$omp end parallel
  else
    stop 1
  end if
end subroutine

! Same scenario but the if-construct has a GOTO exiting the construct, so
! the escape check fires and the wrap is skipped — the original
! unstructured CFG with fir.unreachable is emitted.
!CHECK-LABEL: func.func @_QPsimple1_goto
!CHECK-NOT:   scf.execute_region
!CHECK:   cf.cond_br %{{[0-9]+}}, ^bb[[THEN:[0-9]+]], ^bb[[ELSE:[0-9]+]]
!CHECK: ^bb[[THEN]]:
!CHECK:   omp.parallel
!CHECK:   cf.br ^bb[[ENDIF:[0-9]+]]
!CHECK: ^bb[[ELSE]]:
!CHECK:   fir.call @_FortranAStopStatement
!CHECK:   fir.unreachable
!CHECK: ^bb[[ENDIF]]:
!CHECK:   return

subroutine simple1_goto(y)
  implicit none
  logical, intent(in) :: y
  integer :: i
  if (y) then
    !$omp parallel
    i = 1
    !$omp end parallel
    goto 100
  else
    stop 1
  end if
  i = 2
100 continue
end subroutine

!CHECK-LABEL: func.func @_QPsimple2
!CHECK:   scf.execute_region
!CHECK:   cf.cond_br %{{[0-9]+}}, ^bb[[THEN:[0-9]+]], ^bb[[ELSE:[0-9]+]]
!CHECK: ^bb[[THEN]]:
!CHECK:   omp.parallel
!CHECK:   cf.br ^bb[[MERGE:[0-9]+]]
!CHECK: ^bb[[ELSE]]:
!CHECK:   fir.call @_FortranAStopStatement
!CHECK:   fir.unreachable
!CHECK: ^bb[[MERGE]]:
!CHECK:   scf.yield
!CHECK:   fir.call @_FortranAioOutputReal64
!CHECK:   return
subroutine simple2(x, yn)
  implicit none
  logical, intent(in) :: yn
  integer, intent(in) :: x
  integer :: i
  real(8) :: E
  E = 0d0

  if (yn) then
     !$omp parallel do private(i) reduction(+:E)
     do i = 1, x
        E = E + i
     end do
     !$omp end parallel do
  else
     stop 1
  end if
  print *, E
end subroutine

!CHECK-LABEL: func.func @_QPacccase
!CHECK: fir.select_case %{{[0-9]+}} : i32 [{{.*}}, ^bb[[CASE1:[0-9]+]], {{.*}}, ^bb[[CASE2:[0-9]+]], {{.*}}, ^bb[[CASE3:[0-9]+]]]
!CHECK: ^bb[[CASE1]]:
!CHECK:   acc.serial
!CHECK:   cf.br ^bb[[EXIT:[0-9]+]]
!CHECK: ^bb[[CASE2]]:
!CHECK:   fir.call @_FortranAioOutputAscii
!CHECK:   cf.br ^bb[[EXIT]]
!CHECK: ^bb[[CASE3]]:
!CHECK:   fir.call @_FortranAioOutputAscii
!CHECK:   cf.br ^bb[[EXIT]]
!CHECK: ^bb[[EXIT]]:
!CHECK:   return
subroutine acccase(var)
  integer :: var
  integer :: res(10)
  select case (var)
    case (1)
      print *, "case 1"
      !$acc serial
      res(1) = 1
      !$acc end serial
    case (2)
      print *, "case 2"
    case default
      print *, "case default"
  end select
end subroutine

