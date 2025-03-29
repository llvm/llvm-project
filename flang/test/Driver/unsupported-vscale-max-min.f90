! REQUIRES: x86-registered-target

! RUN: not %flang_fc1 -triple x86_64-unknown-linux-gnu -mvscale-min=1 -mvscale-max=1 -fsyntax-only %s 2>&1 | FileCheck %s

! CHECK: `-mvscale-max` and `-mvscale-min` are not supported for this architecture: x86_64

subroutine func
end subroutine func
