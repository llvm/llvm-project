!RUN: bbc -emit-hlfir -fopenacc -fopenmp -o - %s | FileCheck %s

!https://github.com/llvm/llvm-project/issues/91526

!CHECK-LABEL: func.func @_QPsimple1
!CHECK:   cf.cond_br %{{[0-9]+}}, ^bb[[THEN:[0-9]+]], ^bb[[ELSE:[0-9]+]]
!CHECK: ^bb[[THEN]]:
!CHECK:   omp.parallel
!CHECK:   cf.br ^bb[[ENDIF:[0-9]+]]
!CHECK: ^bb[[ELSE]]:
!CHECK:   fir.call @_FortranAStopStatement
!CHECK:   fir.unreachable
!CHECK: ^bb[[ENDIF]]:
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

!CHECK-LABEL: func.func @_QPsimple2
!CHECK:   cf.cond_br %{{[0-9]+}}, ^bb[[THEN:[0-9]+]], ^bb[[ELSE:[0-9]+]]
!CHECK: ^bb[[THEN]]:
!CHECK:   omp.parallel
!CHECK:   cf.br ^bb[[ENDIF:[0-9]+]]
!CHECK: ^bb[[ELSE]]:
!CHECK:   fir.call @_FortranAStopStatement
!CHECK:   fir.unreachable
!CHECK: ^bb[[ENDIF]]:
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

