! Test various aspects around call lowering. More detailed tests around core
! requirements are done in call-xxx.f90 and dummy-argument-xxx.f90 files.

! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPtest_nested_calls
subroutine test_nested_calls()
  interface
    subroutine foo(i)
      integer :: i
    end subroutine
    integer function bar()
    end function
  end interface
  ! CHECK: %[[result_storage:.*]] = fir.alloca i32 {adapt.valuebyref}
  ! CHECK: %[[result:.*]] = fir.call @_QPbar() {{.*}}: () -> i32
  ! CHECK: fir.store %[[result]] to %[[result_storage]] : !fir.ref<i32>
  ! CHECK: fir.call @_QPfoo(%[[result_storage]]) {{.*}}: (!fir.ref<i32>) -> ()
  call foo(bar())
end subroutine

! Check correct lowering of the result from call to bind(c) function that
! return a char.
subroutine call_bindc_char()
  interface
  function int_to_char(int) bind(c)
    use iso_c_binding
    character(kind=c_char) :: int_to_char
    integer(c_int), value :: int
  end function
  end interface

  print*, int_to_char(40)
end subroutine
! CHECK-LABEL: func.func @_QPcall_bindc_char
! CHECK: %{{.*}} = fir.call @int_to_char(%{{.*}}) {{.*}}: (i32) -> !fir.char<1>

! Check correct lowering of function body that return char and have the bind(c)
! attribute.
function f_int_to_char(i) bind(c, name="f_int_to_char")
  use iso_c_binding
  character(kind=c_char) :: f_int_to_char
  integer(c_int), value :: i
  f_int_to_char = char(i)
end function

! CHECK-LABEL: func.func @f_int_to_char(
! CHECK-SAME: %[[ARG0:.*]]: i32 {fir.bindc_name = "i"}) -> !fir.char<1> attributes {fir.bindc_name = "f_int_to_char"} {
! CHECK: %[[CHARBOX:.*]] = fir.alloca !fir.char<1> {adapt.valuebyref}
! CHECK: %[[INT_I:.*]] = fir.alloca i32
! CHECK: fir.store %[[ARG0]] to %[[INT_I]] : !fir.ref<i32>
! CHECK: %[[RESULT:.*]] = fir.alloca !fir.char<1> {bindc_name = "f_int_to_char", uniq_name = "_QFf_int_to_charEf_int_to_char"}
! CHECK: %[[ARG0_2:.*]] = fir.load %[[INT_I]] : !fir.ref<i32>
! CHECK: %[[ARG0_I64:.*]] = fir.convert %[[ARG0_2]] : (i32) -> i64
! CHECK: %[[ARG0_I8:.*]] = fir.convert %[[ARG0_I64]] : (i64) -> i8
! CHECK: %[[UNDEF:.*]] = fir.undefined !fir.char<1>
! CHECK: %[[CHAR_RES:.*]] = fir.insert_value %[[UNDEF]], %[[ARG0_I8]], [0 : index] : (!fir.char<1>, i8) -> !fir.char<1>
! CHECK: fir.store %[[CHAR_RES]] to %[[CHARBOX]] : !fir.ref<!fir.char<1>>
! CHECK: %[[LOAD_CHARBOX:.*]] = fir.load %[[CHARBOX]] : !fir.ref<!fir.char<1>>
! CHECK: fir.store %[[LOAD_CHARBOX]] to %[[RESULT]] : !fir.ref<!fir.char<1>>
! CHECK: %[[LOAD_RES:.*]] = fir.load %[[RESULT]] : !fir.ref<!fir.char<1>>
! CHECK: return %[[LOAD_RES]] : !fir.char<1>
