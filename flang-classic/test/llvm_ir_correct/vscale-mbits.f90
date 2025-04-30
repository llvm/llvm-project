! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

!** Test checking msv-vector-bits are passed correctly
! REQUIRES: aarch64-registered-target
! REQUIRES: llvm-13

! RUN: %flang -S -emit-llvm -target aarch64-linux-gnu -march=armv8-a+sve -msve-vector-bits=128 %s -o - | FileCheck %s -check-prefix=ATTRS-SVE-128
! RUN: %flang -S -emit-llvm -target aarch64-linux-gnu -march=armv8-a+sve -msve-vector-bits=128+ %s -o - | FileCheck %s -check-prefix=ATTRS-SVE-128PLUS
! RUN: %flang -S -emit-llvm -target aarch64-linux-gnu -march=armv8-a+sve -msve-vector-bits=256 %s -o - | FileCheck %s -check-prefix=ATTRS-SVE-256
! RUN: %flang -S -emit-llvm -target aarch64-linux-gnu -march=armv8-a+sve -msve-vector-bits=256+ %s -o - | FileCheck %s -check-prefix=ATTRS-SVE-256PLUS
! RUN: %flang -S -emit-llvm -target aarch64-linux-gnu -march=armv8-a+sve2 -msve-vector-bits=512 %s -o - | FileCheck %s -check-prefix=ATTRS-SVE2-512
! RUN: %flang -S -emit-llvm -target aarch64-linux-gnu -march=armv8-a+sve2 -msve-vector-bits=512+ %s -o - | FileCheck %s -check-prefix=ATTRS-SVE2-512PLUS
! RUN: %flang -S -emit-llvm -target aarch64-linux-gnu -march=armv8-a+sve2-sha3 -msve-vector-bits=2048 %s -o - | FileCheck %s -check-prefix=ATTRS-SVE2SHA3-2048
! RUN: %flang -S -emit-llvm -target aarch64-linux-gnu -march=armv8-a+sve2-sha3 -msve-vector-bits=2048+ %s -o - | FileCheck %s -check-prefix=ATTRS-SVE2SHA3-2048PLUS
! RUN: %flang -S -emit-llvm -target aarch64-linux-gnu -march=armv8-a+sve2 -msve-vector-bits=scalable %s -o - | FileCheck %s -check-prefix=ATTRS-SVE2-SCALABLE
      program tz
       integer :: i
       integer :: acc(100)
       do i = 1, 100
            acc(i) = 5
       end do
       print *, acc(100)
      end program
! ATTRS-SVE-128: attributes #{{[0-9]+}}
! ATTRS-SVE-128-DAG: "target-features"="+neon{{(,\+v8a)*}},+sve"
! ATTRS-SVE-128-DAG: vscale_range(1,1)
! ATTRS-SVE-128PLUS: attributes #{{[0-9]+}}
! ATTRS-SVE-128PLUS-DAG: "target-features"="+neon{{(,\+v8a)*}},+sve"
! ATTRS-SVE-128PLUS-DAG: vscale_range(1,0)
! ATTRS-SVE-256: attributes #{{[0-9]+}}
! ATTRS-SVE-256-DAG: "target-features"="+neon{{(,\+v8a)*}},+sve"
! ATTRS-SVE-256-DAG: vscale_range(2,2)
! ATTRS-SVE-256PLUS: attributes #{{[0-9]+}}
! ATTRS-SVE-256PLUS-DAG: "target-features"="+neon{{(,\+v8a)*}},+sve"
! ATTRS-SVE-256PLUS-DAG: vscale_range(2,0)
! ATTRS-SVE2-512: attributes #{{[0-9]+}}
! ATTRS-SVE2-512-DAG: "target-features"="+neon{{(,\+v8a)*}},+sve2,+sve"
! ATTRS-SVE2-512-DAG: vscale_range(4,4)
! ATTRS-SVE2-512PLUS: attributes #{{[0-9]+}}
! ATTRS-SVE2-512PLUS-DAG: "target-features"="+neon{{(,\+v8a)*}},+sve2,+sve"
! ATTRS-SVE2-512PLUS-DAG: vscale_range(4,0)
! ATTRS-SVE2SHA3-2048: attributes #{{[0-9]+}}
! ATTRS-SVE2SHA3-2048-DAG: "target-features"="+neon{{(,\+v8a)*}},+sve2-sha3,+sve,+sve2"
! ATTRS-SVE2SHA3-2048-DAG: vscale_range(16,16)
! ATTRS-SVE2SHA3-2048PLUS: attributes #{{[0-9]+}}
! ATTRS-SVE2SHA3-2048PLUS-DAG: "target-features"="+neon{{(,\+v8a)*}},+sve2-sha3,+sve,+sve2"
! ATTRS-SVE2SHA3-2048PLUS-DAG: vscale_range(16,0)
! ATTRS-SVE2-SCALABLE: attributes #{{[0-9]+}}
! ATTRS-SVE2-SCALABLE-DAG: "target-features"="+neon{{(,\+v8a)*}},+sve2,+sve"
! ATTRS-SVE2-SCALABLE-DAG: vscale_range(1,16)
