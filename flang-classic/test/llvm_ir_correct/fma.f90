! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

! Test for fp-contract and fma flags
! REQUIRES: aarch64-registered-target

! RUN: %flang -target aarch64-linux-gnu -Kieee -O1 -c -S -emit-llvm %s -o - | FileCheck %s -check-prefix=PRESENCE
! RUN: %flang -target aarch64-linux-gnu -Kieee -O1 -c -S -emit-llvm -fma %s -o - | FileCheck %s -check-prefix=PRESENCE
! RUN: %flang -target aarch64-linux-gnu -Kieee -O1 -c -S -emit-llvm -ffp-contract=on %s -o - | FileCheck %s -check-prefix=PRESENCE
! RUN: %flang -target aarch64-linux-gnu -Kieee -O1 -c -S -emit-llvm -ffp-contract=fast %s -o - | FileCheck %s -check-prefix=PRESENCE

! RUN: %flang -target aarch64-linux-gnu -Kieee -O1 -c -S -emit-llvm -ffp-contract=off %s -o - | FileCheck %s -check-prefix=ABSENCE
! RUN: %flang -target aarch64-linux-gnu -Kieee -O1 -c -S -emit-llvm -nofma %s -o - | FileCheck %s -check-prefix=ABSENCE

! RUN: %flang -target aarch64-linux-gnu -Kieee -c -S -emit-llvm -ffp-contract=on %s -o - | FileCheck %s -check-prefix=ABSENCE
! RUN: %flang -target aarch64-linux-gnu -Kieee -c -S -emit-llvm -ffp-contract=fast %s -o - | FileCheck %s -check-prefix=ABSENCE

! RUN: %flang -target aarch64-linux-gnu -Kieee -c -S -emit-llvm -fma %s -o - | FileCheck %s -check-prefix=ABSENCE
! RUN: %flang -target aarch64-linux-gnu -Kieee -c -S -emit-llvm %s -o - | FileCheck %s -check-prefix=ABSENCE

function fn(x,y,z)
  real :: x,y,z
  fn = x * y + z
end function

! PRESENCE:  {{.*}}@fn_{{.*}}
! PRESENCE:  {{.*}}fmuladd{{.*}}

! ABSENCE: {{.*}}@fn_{{.*}}
! ABSENCE-NOT:  {{.*}}fmuladd{{.*}}
