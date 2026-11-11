! RUN: %flang_fc1 -emit-hlfir -mmlir --mlir-print-debuginfo -mmlir --mlir-print-local-scope -o - %s | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -mmlir --mlir-print-debuginfo -mmlir --mlir-print-local-scope -mmlir --lower-do-while-to-scf-while -o - %s | FileCheck %s --check-prefix=SCFWHILE

! Source locations of the scf.execute_region wrapping an unstructured
! construct and of the code ending a DO construct. The region entry is at the
! construct's opening statement and its exit at the construct's END statement,
! so that stepping through the generated code does not go back to a previous
! line.

! CHECK-LABEL: func.func @_QPif_first
! CHECK:         scf.yield loc("{{.*}}":[[@LINE+4]]:3)
! CHECK-NEXT:  } loc("{{.*}}":[[@LINE+3]]:3)
subroutine if_first(var)
  integer :: var
  if (var .ne. 1) stop 1
  call foo()
end subroutine

! CHECK-LABEL: func.func @_QPif_stmt
! CHECK:         scf.yield loc("{{.*}}":[[@LINE+5]]:3)
! CHECK-NEXT:  } loc("{{.*}}":[[@LINE+4]]:3)
subroutine if_stmt(var)
  integer :: var
  call foo()
  if (var .ne. 1) stop 1
  call foo()
end subroutine

! CHECK-LABEL: func.func @_QPif_construct
! CHECK:         scf.yield loc("{{.*}}":[[@LINE+8]]:3)
! CHECK-NEXT:  } loc("{{.*}}":[[@LINE+4]]:3)
subroutine if_construct(var)
  integer :: var
  call foo()
  if (var .ne. 1) then
    call foo()
    stop 1
  end if
  call foo()
end subroutine

! CHECK-LABEL: func.func @_QPif_in_do
! CHECK:         fir.do_loop
! CHECK:           scf.yield loc("{{.*}}":[[@LINE+10]]:5)
! CHECK-NEXT:    } loc("{{.*}}":[[@LINE+6]]:5)
! CHECK:         } loc("{{.*}}":[[@LINE+4]]:3)
! CHECK-NEXT:    fir.convert {{.*}} loc("{{.*}}":[[@LINE+9]]:3)
subroutine if_in_do(n)
  integer :: n, i
  do i = 1, n
    if (i .eq. n) then
      call foo()
      stop 2
    end if
    call foo()
  end do
end subroutine

! CHECK-LABEL: func.func @_QPdo_exit
! CHECK:         scf.execute_region
! CHECK:           arith.addi {{.*}} loc("{{.*}}":[[@LINE+11]]:3)
! CHECK-NEXT:      fir.store {{.*}} loc("{{.*}}":[[@LINE+10]]:3)
! CHECK-NEXT:      cf.br ^bb{{[0-9]+}} loc("{{.*}}":[[@LINE+9]]:3)
! CHECK:           scf.yield loc("{{.*}}":[[@LINE+8]]:3)
! CHECK-NEXT:    } loc("{{.*}}":[[@LINE+4]]:3)
subroutine do_exit(n)
  integer :: n, i
  call foo()
  do i = 1, n
    call foo()
    if (i .eq. 5) exit
  end do
  call foo()
end subroutine

! CHECK-LABEL: func.func @_QPdo_while
! CHECK:         scf.execute_region
! CHECK:           cf.br ^bb1 loc("{{.*}}":[[@LINE+9]]:3)
! CHECK:           scf.yield loc("{{.*}}":[[@LINE+8]]:3)
! CHECK-NEXT:    } loc("{{.*}}":[[@LINE+4]]:3)
subroutine do_while(n)
  integer :: n
  call foo()
  do while (n .gt. 0)
    n = n - 1
    if (n .eq. 5) exit
  end do
  call foo()
end subroutine

! CHECK-LABEL: func.func @_QPdo_body
! CHECK:         fir.do_loop
! CHECK:           scf.execute_region
! CHECK:             scf.yield loc("{{.*}}":[[@LINE+12]]:3)
! CHECK-NEXT:      } loc("{{.*}}":[[@LINE+4]]:3)
subroutine do_body(n, a)
  integer :: n, i
  real :: a(n)
  do i = 1, n
    if (a(i) > 0.0) then
      a(i) = 1.0
      goto 90
    end if
    a(i) = 2.0
90  continue
  end do
end subroutine

! SCFWHILE-LABEL: func.func @_QPdo_while_structured
! SCFWHILE:         scf.while
! SCFWHILE:           scf.condition({{.*}}) loc("{{.*}}":[[@LINE+5]]:3)
! SCFWHILE:           scf.yield loc("{{.*}}":[[@LINE+6]]:3)
! SCFWHILE-NEXT:    } loc("{{.*}}":[[@LINE+3]]:3)
subroutine do_while_structured(n)
  integer :: n
  do while (n .gt. 0)
    n = n - 1
  end do
end subroutine
