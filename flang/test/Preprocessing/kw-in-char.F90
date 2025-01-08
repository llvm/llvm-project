! RUN: %flang -E %s 2>&1 | FileCheck %s
! CHECK: subroutine test_b_wrapper_c() bind(C, name="test_b_c_f")
#define TEMP_LETTER b
#define VAL c
subroutine test_&
TEMP_LETTER&
_wrapper_&
VAL&
 () bind(C, name="test_&
    &TEMP_LETTER&
    &_&
    &VAL&
    &_f")
end
