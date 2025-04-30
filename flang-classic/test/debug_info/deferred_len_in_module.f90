!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

!RUN: %flang -g -S -emit-llvm %s -o - | FileCheck %s

!CHECK-DAG: distinct !DIGlobalVariable(name: "def_arr"{{.*}}, type: ![[DERIVEDSTRING:[0-9]+]]
!CHECK-DAG: ![[DERIVEDSTRING]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[STRING:[0-9]+]]
!CHECK-DAG: ![[STRING]] = distinct !DIStringType(name: "character(*)", stringLength: ![[STRING_LEN:[0-9]+]]
!CHECK-DAG: ![[STRING_LEN]] = distinct !DIGlobalVariable(scope: ![[MODULE:[0-9]+]]
!CHECK-DAG: ![[MODULE]] = !DIModule{{.*}}, name: "samp"


module samp
    CHARACTER(len=:), ALLOCATABLE :: def_arr
    CHARACTER(len=:), ALLOCATABLE :: def_arr2
end module samp
