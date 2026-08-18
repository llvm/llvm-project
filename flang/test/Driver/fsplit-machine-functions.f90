! Test handling of -fsplit-machine-functions and -fno-split-machine-functions.

! RUN: %if x86-registered-target %{ %flang_fc1 -emit-llvm -triple x86_64-unknown-linux-gnu %s -o - | FileCheck %s --check-prefix=NEG_FLAG %}
! RUN: %if x86-registered-target %{ %flang -### --target=x86_64-unknown-linux-gnu -fsplit-machine-functions %s 2>&1 | FileCheck %s --check-prefix=POS_FLAG %}
! RUN: %if x86-registered-target %{ %flang -### --target=x86_64-unknown-linux-gnu -fno-split-machine-functions %s 2>&1 | FileCheck %s --check-prefix=NEG_FLAG %}
! RUN: %if x86-registered-target %{ %flang -### --target=x86_64-unknown-linux-gnu -fsplit-machine-functions -fno-split-machine-functions %s 2>&1 | FileCheck %s --check-prefix=NEG_FLAG %}
! RUN: %if x86-registered-target %{ %flang -### --target=x86_64-unknown-linux-gnu -fno-split-machine-functions -fsplit-machine-functions %s 2>&1 | FileCheck %s --check-prefix=POS_FLAG %}
! RUN: %if arm-registered-target %{ not %flang -### --target=arm-unknown-linux-gnueabi -fsplit-machine-functions %s 2>&1 | FileCheck %s --check-prefix=CHECK_ERROR %}

! POS_FLAG: "-fsplit-machine-functions"
! NEG_FLAG-NOT: "-fsplit-machine-functions"
! CHECK_ERROR: error: unsupported option '-fsplit-machine-functions' for target

subroutine test(x)
    integer, intent(in) :: x
end subroutine test
