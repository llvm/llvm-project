!===----------------------------------------------------------------------===!
! This directory can be used to add Integration tests involving multiple
! stages of the compiler (for eg. from Fortran to LLVM IR). It should not
! contain executable tests. We should only add tests here sparingly and only
! if there is no other way to test. Repeat this message in each test that is
! added to this directory and sub-directories.
!===----------------------------------------------------------------------===!

!RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 %s -o - | tco -test-gen | FileCheck %s
!RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -fopenmp-is-device %s -o - | tco -test-gen | FileCheck %s

program main
    use, intrinsic ::  iso_c_binding
    implicit none
    interface
    subroutine myinit(priv, orig) bind(c,name="myinit")
        use, intrinsic :: iso_c_binding
        implicit none
        integer::priv, orig
    end subroutine myinit

    function mycombine(lhs, rhs) bind(c,name="mycombine")
        use, intrinsic :: iso_c_binding
        implicit none
        integer::lhs, rhs, mycombine
    end function mycombine
 end interface
     !$omp declare reduction(myreduction:integer:omp_out = mycombine(omp_out, omp_in)) initializer(myinit(omp_priv, omp_orig))

    integer :: i, s, a(10)
    !$omp target
    s = 0
    !$omp do reduction(myreduction:s)
    do i = 1, 10
       s = mycombine(s, a(i))
    enddo
    !$omp end do
    !$omp end target
 end program main

!CHECK: llvm.func @myinit(!llvm.ptr, !llvm.ptr)
!CHECK-SAME: {{.*}}, omp.declare_target = #omp.declaretarget<device_type = any, capture_clause = to, implicit = true>{{.*}}
!CHECK-LABEL: llvm.func @mycombine(!llvm.ptr, !llvm.ptr)
!CHECK-SAME: {{.*}}, omp.declare_target = #omp.declaretarget<device_type = any, capture_clause = to, implicit = true>{{.*}}
