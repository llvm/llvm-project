!RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s --check-prefixes ALL,HOST
!RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-is-device %s -o - | FileCheck %s --check-prefixes ALL

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

!HOST-LABEL: func.func {{.*}} @host_interface()
!HOST-SAME: {{.*}}, omp.declare_target = #omp.declaretarget<device_type = (host), capture_clause = (enter)>{{.*}}
!ALL-LABEL: func.func {{.*}} @called_from_target_interface(!fir.ref<i64>, !fir.ref<i64>)
!ALL-SAME: {{.*}}, omp.declare_target = #omp.declaretarget<device_type = (nohost), capture_clause = (to)>{{.*}}
!ALL-LABEL: func.func {{.*}} @any_interface()
!ALL-SAME: {{.*}}, omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (enter)>{{.*}}
!ALL-LABEL: func.func {{.*}} @device_interface()
!ALL-SAME: {{.*}}, omp.declare_target = #omp.declaretarget<device_type = (nohost), capture_clause = (enter)>{{.*}}
