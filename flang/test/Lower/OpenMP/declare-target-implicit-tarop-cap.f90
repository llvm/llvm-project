!RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 %s -o - | FileCheck %s
!RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -fopenmp-is-device %s -o - | FileCheck %s  --check-prefix=DEVICE
!RUN: bbc -emit-hlfir -fopenmp -fopenmp-version=52 %s -o - | FileCheck %s
!RUN: bbc -emit-hlfir -fopenmp -fopenmp-version=52 -fopenmp-is-target-device %s -o - | FileCheck %s --check-prefix=DEVICE

! DEVICE-LABEL: func.func @_QPimplicit_capture
! DEVICE-SAME: {{.*}}attributes {omp.declare_target = #omp.declaretarget<device_type = (nohost), capture_clause = (to)>{{.*}}}
function implicit_capture() result(i)
   implicit none
   integer :: i
   i = 1
end function implicit_capture

subroutine subr_target()
   integer :: n
!$omp target map(tofrom:n)
   n = implicit_capture()
!$omp end target
end subroutine

!! -----

! CHECK-LABEL: func.func @_QPimplicitly_captured_nest_twice
! CHECK-SAME: {{.*}}attributes {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (to)>{{.*}}}
function implicitly_captured_nest_twice() result(i)
   integer :: i
   i = 10
end function implicitly_captured_nest_twice

! CHECK-LABEL: func.func @_QPimplicitly_captured_one_twice
! CHECK-SAME: {{.*}}attributes {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (to)>{{.*}}}
function implicitly_captured_one_twice() result(k)
!$omp declare target to(implicitly_captured_one_twice) device_type(host)
   k = implicitly_captured_nest_twice()
end function implicitly_captured_one_twice

! CHECK-LABEL: func.func @_QPimplicitly_captured_nest_twice_enter
! CHECK-SAME: {{.*}}attributes {omp.declare_target = #omp.declaretarget<device_type = (host), capture_clause = (enter)>{{.*}}}
function implicitly_captured_nest_twice_enter() result(i)
   integer :: i
   i = 10
end function implicitly_captured_nest_twice_enter

! CHECK-LABEL: func.func @_QPimplicitly_captured_one_twice_enter
! CHECK-SAME: {{.*}}attributes {omp.declare_target = #omp.declaretarget<device_type = (host), capture_clause = (enter)>{{.*}}}
function implicitly_captured_one_twice_enter() result(k)
!$omp declare target enter(implicitly_captured_one_twice_enter) device_type(host)
   k = implicitly_captured_nest_twice_enter()
end function implicitly_captured_one_twice_enter

! DEVICE-LABEL: func.func @_QPimplicitly_captured_two_twice
! DEVICE-SAME: {{.*}}attributes {omp.declare_target = #omp.declaretarget<device_type = (nohost), capture_clause = (to)>{{.*}}}
function implicitly_captured_two_twice() result(y)
   integer :: y
   y = 5
end function implicitly_captured_two_twice


function target_function_test_device() result(j)
   integer :: i, j
   !$omp target map(tofrom: i, j)
   i = implicitly_captured_one_twice()
   j = implicitly_captured_two_twice() + i
   !$omp end target
end function target_function_test_device

!! -----

! DEVICE-LABEL: func.func @_QPimplicitly_captured_recursive
! DEVICE-SAME: {{.*}}attributes {omp.declare_target = #omp.declaretarget<device_type = (nohost), capture_clause = (to)>{{.*}}}
recursive function implicitly_captured_recursive(increment) result(k)
   integer :: increment, k
   if (increment == 10) then
      k = increment
   else
      k = implicitly_captured_recursive(increment + 1)
   end if
end function implicitly_captured_recursive

function target_function_recurse() result(i)
   integer :: i
   !$omp target map(tofrom: i)
   i = implicitly_captured_recursive(0)
   !$omp end target
end function target_function_recurse
