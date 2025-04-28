! Test that we can disable the ExternalNameConversion pass in flang-new.

! RUN: %flang_fc1 -S %s -o - 2>&1 | FileCheck %s --check-prefix=EXTNAMES
! RUN: %flang_fc1 -S -mmlir -disable-external-name-interop %s -o - 2>&1 | FileCheck %s --check-prefix=INTNAMES

! EXTNAMES: test_ext_names_
! INTNAMES: _QPtest_ext_names
subroutine test_ext_names
end subroutine
