! Test -ffunction-sections and -fdata-sections codegen (X86).

! REQUIRES: x86-registered-target

! RUN: %flang_fc1 -triple x86_64-unknown-linux-gnu -S -ffunction-sections \
! RUN:     -o - %s | FileCheck %s --check-prefix=FUNC-SECT
! RUN: %flang_fc1 -triple x86_64-unknown-linux-gnu -S -o - %s \
! RUN:   | FileCheck %s --check-prefix=FUNC-PLAIN

! RUN: %flang_fc1 -triple x86_64-unknown-linux-gnu -S -fdata-sections \
! RUN:     -o - %s | FileCheck %s --check-prefix=DATA-SECT
! RUN: %flang_fc1 -triple x86_64-unknown-linux-gnu -S -o - %s \
! RUN:   | FileCheck %s --check-prefix=DATA-PLAIN

module data_sect_mod
  integer, save :: g = 1
end module

subroutine foo
end subroutine

program test
  use data_sect_mod
  call foo
end program

! FUNC-SECT: .section{{.*}}.text.
! FUNC-PLAIN-NOT: .section{{.*}}.text.

! DATA-SECT: .section{{.*}}.data.
! DATA-PLAIN: .data
! DATA-PLAIN-NOT: .section{{.*}}.data.
