! RUN: %flang_fc1 -emit-fir -fopenmp -fopenmp-version=51 %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPtest_device_kind_host
! CHECK: fir.call @_QFbase_hostPvsub(){{.*}}: () -> ()
! CHECK-NOT: fir.call @_QPbase_host
subroutine test_device_kind_host
  call base_host()
end subroutine test_device_kind_host

subroutine base_host
  !$omp declare variant (base_host:vsub) match (device={kind(host)})
contains
  subroutine vsub
  end subroutine
end subroutine base_host

! kind(nohost) does not match on a host compilation: the base call is kept.

! CHECK-LABEL: func.func @_QPtest_device_kind_nohost
! CHECK: fir.call @_QPbase_nohost(){{.*}}: () -> ()
! CHECK-NOT: fir.call @_QFbase_nohostPvsub
subroutine test_device_kind_nohost
  call base_nohost()
end subroutine test_device_kind_nohost

subroutine base_nohost
  !$omp declare variant (base_nohost:vsub) match (device={kind(nohost)})
contains
  subroutine vsub
  end subroutine
end subroutine base_nohost

! A device kind that matches neither host nor nohost also does not select
! a variant.

! CHECK-LABEL: func.func @_QPtest_device_no_match
! CHECK: fir.call @_QPbase(){{.*}}: () -> ()
! CHECK: omp.parallel
! CHECK: fir.call @_QPbase(){{.*}}: () -> ()
! CHECK-NOT: fir.call @_QFbasePvsub
subroutine test_device_no_match
  call base()
  !$omp parallel
  call base()
  !$omp end parallel
end subroutine test_device_no_match

subroutine base
  !$omp declare variant (base:vsub) match (device={kind(fpga)})
contains
  subroutine vsub
  end subroutine
end subroutine base
