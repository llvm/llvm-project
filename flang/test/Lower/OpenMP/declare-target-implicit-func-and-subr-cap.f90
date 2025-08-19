!RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 %s -o - | FileCheck %s
!RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 -fopenmp-is-target-device %s -o - | FileCheck %s  --check-prefix=DEVICE
!RUN: bbc -emit-hlfir -fopenmp -fopenmp-version=50 %s -o - | FileCheck %s
!RUN: bbc -emit-hlfir -fopenmp -fopenmp-version=50 -fopenmp-is-target-device %s -o - | FileCheck %s --check-prefix=DEVICE

! CHECK-LABEL: func.func @_QPimplicitly_captured
! CHECK-SAME: {{.*}}attributes {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (to), automap = false>{{.*}}}
function implicitly_captured(toggle) result(k)
   integer :: i, j, k
   logical :: toggle
   i = 10
   j = 5
   if (toggle) then
      k = i
   else
      k = j
   end if
end function implicitly_captured


! CHECK-LABEL: func.func @_QPtarget_function
! CHECK-SAME: {{.*}}attributes {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (to), automap = false>{{.*}}}
function target_function(toggle) result(i)
!$omp declare target
   integer :: i
   logical :: toggle
   i = implicitly_captured(toggle)
end function target_function

!! -----

! CHECK-LABEL: func.func @_QPimplicitly_captured_twice
! CHECK-SAME: {{.*}}attributes {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (to), automap = false>{{.*}}}
function implicitly_captured_twice() result(k)
   integer :: i
   i = 10
   k = i
end function implicitly_captured_twice

! CHECK-LABEL: func.func @_QPtarget_function_twice_host
! CHECK-SAME: {{.*}}attributes {omp.declare_target = #omp.declaretarget<device_type = (host), capture_clause = (to), automap = false>{{.*}}}
function target_function_twice_host() result(i)
!$omp declare target to(target_function_twice_host) device_type(host)
   integer :: i
   i = implicitly_captured_twice()
end function target_function_twice_host

! DEVICE-LABEL: func.func @_QPtarget_function_twice_device
! DEVICE-SAME: {{.*}}attributes {omp.declare_target = #omp.declaretarget<device_type = (nohost), capture_clause = (to), automap = false>{{.*}}}
function target_function_twice_device() result(i)
!$omp declare target to(target_function_twice_device) device_type(nohost)
   integer :: i
   i = implicitly_captured_twice()
end function target_function_twice_device

!! -----

! DEVICE-LABEL: func.func @_QPimplicitly_captured_nest
! DEVICE-SAME: {{.*}}attributes {omp.declare_target = #omp.declaretarget<device_type = (nohost), capture_clause = (to), automap = false>{{.*}}}
function implicitly_captured_nest() result(k)
   integer :: i
   i = 10
   k = i
end function implicitly_captured_nest

! DEVICE-LABEL: func.func @_QPimplicitly_captured_one
! DEVICE-SAME: {{.*}}attributes {omp.declare_target = #omp.declaretarget<device_type = (nohost), capture_clause = (to){{.*}}}
function implicitly_captured_one() result(k)
   k = implicitly_captured_nest()
end function implicitly_captured_one

! DEVICE-LABEL: func.func @_QPimplicitly_captured_two
! DEVICE-SAME: {{.*}}attributes {omp.declare_target = #omp.declaretarget<device_type = (nohost), capture_clause = (to), automap = false>{{.*}}}
function implicitly_captured_two() result(k)
   integer :: i
   i = 10
   k = i
end function implicitly_captured_two

! DEVICE-LABEL: func.func @_QPtarget_function_test
! DEVICE-SAME: {{.*}}attributes {omp.declare_target = #omp.declaretarget<device_type = (nohost), capture_clause = (to), automap = false>{{.*}}}
function target_function_test() result(j)
!$omp declare target to(target_function_test) device_type(nohost)
   integer :: i, j
   i = implicitly_captured_one()
   j = implicitly_captured_two() + i
end function target_function_test

!! -----

! CHECK-LABEL: func.func @_QPimplicitly_captured_nest_twice
! CHECK-SAME: {{.*}}attributes {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (to), automap = false>{{.*}}}
function implicitly_captured_nest_twice() result(k)
   integer :: i
   i = 10
   k = i
end function implicitly_captured_nest_twice

! CHECK-LABEL: func.func @_QPimplicitly_captured_one_twice
! CHECK-SAME: {{.*}}attributes {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (to), automap = false>{{.*}}}
function implicitly_captured_one_twice() result(k)
   k = implicitly_captured_nest_twice()
end function implicitly_captured_one_twice

! CHECK-LABEL: func.func @_QPimplicitly_captured_two_twice
! CHECK-SAME: {{.*}}attributes {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (to), automap = false>{{.*}}}
function implicitly_captured_two_twice() result(k)
   integer :: i
   i = 10
   k = i
end function implicitly_captured_two_twice

! DEVICE-LABEL: func.func @_QPtarget_function_test_device
! DEVICE-SAME: {{.*}}attributes {omp.declare_target = #omp.declaretarget<device_type = (nohost), capture_clause = (to), automap = false>{{.*}}}
function target_function_test_device() result(j)
   !$omp declare target to(target_function_test_device) device_type(nohost)
   integer :: i, j
   i = implicitly_captured_one_twice()
   j = implicitly_captured_two_twice() + i
end function target_function_test_device

! CHECK-LABEL: func.func @_QPtarget_function_test_host
! CHECK-SAME: {{.*}}attributes {omp.declare_target = #omp.declaretarget<device_type = (host), capture_clause = (to), automap = false>{{.*}}}
function target_function_test_host() result(j)
   !$omp declare target to(target_function_test_host) device_type(host)
   integer :: i, j
   i = implicitly_captured_one_twice()
   j = implicitly_captured_two_twice() + i
end function target_function_test_host

!! -----

! DEVICE-LABEL: func.func @_QPimplicitly_captured_with_dev_type_recursive
! DEVICE-SAME: {{.*}}attributes {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (to), automap = false>{{.*}}}
recursive function implicitly_captured_with_dev_type_recursive(increment) result(k)
!$omp declare target to(implicitly_captured_with_dev_type_recursive) device_type(host)
   integer :: increment, k
   if (increment == 10) then
      k = increment
   else
      k = implicitly_captured_with_dev_type_recursive(increment + 1)
   end if
end function implicitly_captured_with_dev_type_recursive

! DEVICE-LABEL: func.func @_QPtarget_function_with_dev_type_recurse
! DEVICE-SAME: {{.*}}attributes {omp.declare_target = #omp.declaretarget<device_type = (nohost), capture_clause = (to), automap = false>{{.*}}}
function target_function_with_dev_type_recurse() result(i)
!$omp declare target to(target_function_with_dev_type_recurse) device_type(nohost)
   integer :: i
   i = implicitly_captured_with_dev_type_recursive(0)
end function target_function_with_dev_type_recurse

!! -----

module test_module
contains
! CHECK-LABEL: func.func @_QMtest_modulePimplicitly_captured_nest_twice
! CHECK-SAME: {{.*}}attributes {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (to), automap = false>{{.*}}}
   function implicitly_captured_nest_twice() result(i)
      integer :: i
      i = 10
   end function implicitly_captured_nest_twice

! CHECK-LABEL: func.func @_QMtest_modulePimplicitly_captured_one_twice
! CHECK-SAME: {{.*}}attributes {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (to), automap = false>{{.*}}}
   function implicitly_captured_one_twice() result(k)
      !$omp declare target to(implicitly_captured_one_twice) device_type(host)
      k = implicitly_captured_nest_twice()
   end function implicitly_captured_one_twice

! DEVICE-LABEL: func.func @_QMtest_modulePimplicitly_captured_two_twice
! DEVICE-SAME: {{.*}}attributes {omp.declare_target = #omp.declaretarget<device_type = (nohost), capture_clause = (to), automap = false>{{.*}}}
   function implicitly_captured_two_twice() result(y)
      integer :: y
      y = 5
   end function implicitly_captured_two_twice

! DEVICE-LABEL: func.func @_QMtest_modulePtarget_function_test_device
! DEVICE-SAME: {{.*}}attributes {omp.declare_target = #omp.declaretarget<device_type = (nohost), capture_clause = (to), automap = false>{{.*}}}
   function target_function_test_device() result(j)
      !$omp declare target to(target_function_test_device) device_type(nohost)
      integer :: i, j
      i = implicitly_captured_one_twice()
      j = implicitly_captured_two_twice() + i
   end function target_function_test_device
end module test_module

!! -----

program mb
   interface
      subroutine caller_recursive
         !$omp declare target to(caller_recursive) device_type(nohost)
      end subroutine

      recursive subroutine implicitly_captured_recursive(increment)
         integer :: increment
      end subroutine
   end interface
end program

! DEVICE-LABEL: func.func @_QPimplicitly_captured_recursive
! DEVICE-SAME: {{.*}}attributes {omp.declare_target = #omp.declaretarget<device_type = (nohost), capture_clause = (to), automap = false>{{.*}}}
recursive subroutine implicitly_captured_recursive(increment)
   integer :: increment
   if (increment == 10) then
      return
   else
      call implicitly_captured_recursive(increment + 1)
   end if
end subroutine

! DEVICE-LABEL: func.func @_QPcaller_recursive
! DEVICE-SAME: {{.*}}attributes {omp.declare_target = #omp.declaretarget<device_type = (nohost), capture_clause = (to), automap = false>{{.*}}}
subroutine caller_recursive
!$omp declare target to(caller_recursive) device_type(nohost)
   call implicitly_captured_recursive(0)
end subroutine
