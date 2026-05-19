!RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 %s -o - | FileCheck %s
!RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -fopenmp-is-device %s -o - | FileCheck %s

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

!CHECK: func.func {{.*}} @myinit(!fir.ref<i32>, !fir.ref<i32>)
!CHECK-SAME: {{.*}}, omp.declare_target = #omp.declaretarget<device_type = (nohost), capture_clause = (to), automap = false>{{.*}}
!CHECK-LABEL: func.func {{.*}} @mycombine(!fir.ref<i32>, !fir.ref<i32>)
!CHECK-SAME: {{.*}}, omp.declare_target = #omp.declaretarget<device_type = (nohost), capture_clause = (to), automap = false>{{.*}}

