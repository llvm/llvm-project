! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

!** Test checking vscale attribute are set correctly
! REQUIRES: aarch64-registered-target
! REQUIRES: llvm-13

! RUN: %flang -S -emit-llvm -target aarch64-linux-gnu -march=armv8-a %s -o - | FileCheck %s -check-prefix=ATTRS-NEON
! RUN: %flang -S -emit-llvm -target aarch64-linux-gnu -march=armv8-a+sve %s -o - | FileCheck %s -check-prefix=ATTRS-SVE
! RUN: %flang -S -emit-llvm -target aarch64-linux-gnu -march=armv8-a+sve2 %s -o - | FileCheck %s -check-prefix=ATTRS-SVE2
! RUN: %flang -S -emit-llvm -target aarch64-linux-gnu -march=armv8-a+sve2-sha3 %s -o - | FileCheck %s -check-prefix=ATTRS-SVE2SHA3
! RUN: %flang -S -emit-llvm -target aarch64-linux-gnu -march=armv8-a+sve+nosve %s -o - | FileCheck %s -check-prefix=ATTRS-SVE-NOSVE
! RUN: %flang -S -emit-llvm -target aarch64-linux-gnu -march=armv8-a+sve2+nosve2-sha3 %s -o - | FileCheck %s -check-prefix=ATTRS-SVE2-NOSVE2SHA3
! RUN: %flang -S -emit-llvm -target aarch64-linux-gnu -march=armv8-a+sve2-sha3+nosve2 %s -o - | FileCheck %s -check-prefix=ATTRS-SVE2SHA3-NOSVE2
! RUN: %flang -S -emit-llvm -target aarch64-linux-gnu -march=armv8-a+sve2-sha3+nosve %s -o - | FileCheck %s -check-prefix=ATTRS-SVE2SHA3-NOSVE
      program tz
       integer :: i
       integer :: acc(100)
       do i = 1, 100
            acc(i) = 5
       end do
       print *, acc(100)
      end program
! ATTRS-NEON-NOT: vscale_range
! ATTRS-NEON: attributes{{.*}}"target-features"="+neon{{(,\+v8a)*}}"
! ATTRS-SVE: attributes #{{[0-9]+}}
! ATTRS-SVE-DAG: "target-features"="+neon{{(,\+v8a)*}},+sve"
! ATTRS-SVE-DAG: vscale_range(1,16)
! ATTRS-SVE2: attributes #{{[0-9]+}}
! ATTRS-SVE2-DAG: "target-features"="+neon{{(,\+v8a)*}},+sve2,+sve"
! ATTRS-SVE2-DAG: vscale_range(1,16)
! ATTRS-SVE2SHA3: attributes #{{[0-9]+}}
! ATTRS-SVE2SHA3-DAG: "target-features"="+neon{{(,\+v8a)*}},+sve2-sha3,+sve,+sve2"
! ATTRS-SVE2SHA3-DAG: vscale_range(1,16)
! ATTRS-SVE-NOSVE-NOT: vscale_range
! ATTRS-SVE-NOSVE: attributes{{.*}}"target-features"="+neon{{(,\+v8a)*}},-sve,-sve2,-sve2-bitperm,-sve2-sha3,-sve2-aes,-sve2-sm4"
! ATTRS-SVE2-NOSVE2SHA3: attributes #{{[0-9]+}}
! ATTRS-SVE2-NOSVE2SHA3-DAG: "target-features"="+neon{{(,\+v8a)*}},+sve2,+sve,-sve2-sha3"
! ATTRS-SVE2-NOSVE2SHA3-DAG: vscale_range(1,16)
! ATTRS-SVE2SHA3-NOSVE2: attributes #{{[0-9]+}}
! ATTRS-SVE2SHA3-NOSVE2-DAG: "target-features"="+neon{{(,\+v8a)*}},+sve,-sve2,-sve2-bitperm,-sve2-sha3,-sve2-aes,-sve2-sm4"
! ATTRS-SVE2SHA3-NOSVE2-DAG: vscale_range(1,16)
! ATTRS-SVE2SHA3-NOSVE-NOT: vscale_range
! ATTRS-SVE2SHA3-NOSVE: attributes{{.*}}"target-features"="+neon{{(,\+v8a)*}},-sve,-sve2,-sve2-bitperm,-sve2-sha3,-sve2-aes,-sve2-sm4"
