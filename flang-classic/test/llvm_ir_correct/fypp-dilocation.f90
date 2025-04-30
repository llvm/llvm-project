! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

! RUN: %flang -cpp -g -S -emit-llvm %s -o - | FileCheck %s

# 111 "fypp-dilocation.F"
function fypp1()
# 111 "fypp-dilocation.F"
  implicit none
# 111 "fypp-dilocation.F"
  integer :: fypp1
# 111 "fypp-dilocation.F"
  fypp1 = 0
# 111 "fypp-dilocation.F"
end function fypp1
# 111 "fypp-dilocation.F"
function fypp2(data)
# 111 "fypp-dilocation.F"
  implicit none
# 111 "fypp-dilocation.F"
  integer :: fypp2
# 111 "fypp-dilocation.F"
  integer, dimension(2), intent(in) :: data
! CHECK: br {{.*}}, !llvm.loop ![[LOOP:[0-9]+]]
# 111 "fypp-dilocation.F"
  fypp2 = sum(data)
# 111 "fypp-dilocation.F"
end function fypp2
# 111 "fypp-dilocation.F"

! CHECK-DAG: ![[DILOC:[0-9]+]] = !DILocation(line: 111, column: {{[0-9]+}}, scope: !{{[0-9]+}})
! CHECK-DAG: !{![[LOOP]], ![[DILOC]], ![[DILOC]]}
