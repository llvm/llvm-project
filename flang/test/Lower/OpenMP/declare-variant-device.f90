! RUN: %flang_fc1 -emit-fir -fopenmp -fopenmp-version=51 %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPtest_device_kind_host
! CHECK: fir.call @_QPvhost(){{.*}}: () -> ()
! CHECK-NOT: fir.call @_QPbase_host
subroutine test_device_kind_host
  call base_host()
end subroutine test_device_kind_host

subroutine base_host
  interface
    subroutine vhost()
    end subroutine
  end interface
  !$omp declare variant (base_host:vhost) match (device={kind(host)})
end subroutine base_host

! kind(nohost) does not match on a host compilation: the base call is kept.

! CHECK-LABEL: func.func @_QPtest_device_kind_nohost
! CHECK: fir.call @_QPbase_nohost(){{.*}}: () -> ()
! CHECK-NOT: fir.call @_QPvnohost
subroutine test_device_kind_nohost
  call base_nohost()
end subroutine test_device_kind_nohost

subroutine base_nohost
  interface
    subroutine vnohost()
    end subroutine
  end interface
  !$omp declare variant (base_nohost:vnohost) match (device={kind(nohost)})
end subroutine base_nohost

! A device kind that matches neither host nor nohost also does not select
! a variant.

! CHECK-LABEL: func.func @_QPtest_device_no_match
! CHECK: fir.call @_QPbase_fpga(){{.*}}: () -> ()
! CHECK: omp.parallel
! CHECK: fir.call @_QPbase_fpga(){{.*}}: () -> ()
! CHECK-NOT: fir.call @_QPvfpga
subroutine test_device_no_match
  call base_fpga()
  !$omp parallel
  call base_fpga()
  !$omp end parallel
end subroutine test_device_no_match

subroutine base_fpga
  interface
    subroutine vfpga()
    end subroutine
  end interface
  !$omp declare variant (base_fpga:vfpga) match (device={kind(fpga)})
end subroutine base_fpga
