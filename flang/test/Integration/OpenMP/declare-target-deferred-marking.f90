!===----------------------------------------------------------------------===!
! This directory can be used to add Integration tests involving multiple
! stages of the compiler (for eg. from Fortran to LLVM IR). It should not
! contain executable tests. We should only add tests here sparingly and only
! if there is no other way to test. Repeat this message in each test that is
! added to this directory and sub-directories.
!===----------------------------------------------------------------------===!

!RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 %s -o - | tco -test-gen | FileCheck %s --check-prefixes ALL,HOST
!RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -fopenmp-is-device %s -o - | tco -test-gen | FileCheck %s --check-prefixes ALL

program main
    use, intrinsic ::  iso_c_binding
    implicit none
    interface
    subroutine any_interface()  bind(c,name="any_interface")
        use, intrinsic :: iso_c_binding
        implicit none
    !$omp declare target enter(any_interface) device_type(any)
    end subroutine any_interface

    subroutine host_interface()  bind(c,name="host_interface")
      use, intrinsic :: iso_c_binding
      implicit none
   !$omp declare target enter(host_interface) device_type(host)
    end subroutine host_interface

    subroutine device_interface()  bind(c,name="device_interface")
        use, intrinsic :: iso_c_binding
        implicit none
    !$omp declare target enter(device_interface) device_type(nohost)
    end subroutine device_interface

    subroutine called_from_target_interface(f1, f2) bind(c,name="called_from_target_interface")
        use, intrinsic :: iso_c_binding
        implicit none
        type(c_funptr),value :: f1
        type(c_funptr),value :: f2
    end subroutine called_from_target_interface

    subroutine called_from_host_interface(f1) bind(c,name="called_from_host_interface")
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_funptr),value :: f1
    end subroutine called_from_host_interface

    subroutine unused_unemitted_interface()  bind(c,name="unused_unemitted_interface")
      use, intrinsic :: iso_c_binding
      implicit none
    !$omp declare target enter(unused_unemitted_interface) device_type(nohost)
    end subroutine unused_unemitted_interface

    end interface

    CALL called_from_host_interface(c_funloc(host_interface))
!$omp target
    CALL called_from_target_interface(c_funloc(any_interface), c_funloc(device_interface))
!$omp end target
 end program main

!HOST-LABEL: llvm.func @host_interface()
!HOST-SAME: {{.*}}, omp.declare_target = #omp.declaretarget<device_type = (host), capture_clause = (enter)>{{.*}}
!ALL-LABEL: llvm.func @called_from_target_interface(!llvm.ptr, !llvm.ptr)
!ALL-SAME: {{.*}}, omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (to), implicit = true>{{.*}}
!ALL-LABEL: llvm.func @any_interface()
!ALL-SAME: {{.*}}, omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (enter)>{{.*}}
!ALL-LABEL: llvm.func @device_interface()
!ALL-SAME: {{.*}}, omp.declare_target = #omp.declaretarget<device_type = (nohost), capture_clause = (enter)>{{.*}}
