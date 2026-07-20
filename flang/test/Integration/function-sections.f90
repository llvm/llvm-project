! Test -ffunction-sections and -fdata-sections codegen.

! DEFINE: %{triple} =
! DEFINE: %{check-func-sect} = %flang_fc1 -triple %{triple} -S -ffunction-sections -o - %s | FileCheck %s --check-prefix=FUNC-SECT
! DEFINE: %{check-func-plain} = %flang_fc1 -triple %{triple} -S -o - %s | FileCheck %s --check-prefix=FUNC-PLAIN
! DEFINE: %{check-data-sect} = %flang_fc1 -triple %{triple} -S -fdata-sections -o - %s | FileCheck %s --check-prefix=DATA-SECT --check-prefix=RODATA-SECT
! DEFINE: %{check-data-plain} = %flang_fc1 -triple %{triple} -S -o - %s | FileCheck %s --check-prefix=DATA-PLAIN --check-prefix=RODATA-PLAIN

! REDEFINE: %{triple} = aarch64-unknown-linux-gnu
! RUN: %if aarch64-registered-target %{ %{check-func-sect} %}
! RUN: %if aarch64-registered-target %{ %{check-func-plain} %}
! RUN: %if aarch64-registered-target %{ %{check-data-sect} %}
! RUN: %if aarch64-registered-target %{ %{check-data-plain} %}

! REDEFINE: %{triple} = x86_64-unknown-linux-gnu
! RUN: %if x86-registered-target %{ %{check-func-sect} %}
! RUN: %if x86-registered-target %{ %{check-func-plain} %}
! RUN: %if x86-registered-target %{ %{check-data-sect} %}
! RUN: %if x86-registered-target %{ %{check-data-plain} %}

module data_sect_mod
  integer, save :: g = 1
end module

module param_mod
  integer, parameter :: n = 42
end module

subroutine foo
end subroutine

program test
  use data_sect_mod
  use param_mod
  integer :: x
  call foo
  x = n
end program

! FUNC-SECT: .section{{.*}}.text.foo_
! FUNC-PLAIN-NOT: .section{{.*}}.text.

! DATA-SECT: .section{{.*}}.data._QMdata_sect_modEg
! DATA-PLAIN: .data
! DATA-PLAIN-NOT: .section{{.*}}.data.

! RODATA-SECT: .section{{.*}}.rodata._QMparam_modECn
! RODATA-PLAIN-NOT: .section{{.*}}.rodata._QMparam_modECn
