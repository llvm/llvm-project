!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Debug flags: q,0,1 => stdio dump; q,11,16 => dump ILI just before LLVM IR conversion.
! RUN: %flang -c -Mq,0,1 -Mq,11,16 %s 2>&1 | FileCheck %s

! Test that the ILT dump contains a BIH with the HEAD bit set.
! This is used for finding loop backedges to mark them with loop metadata.

! 0. Find subroutine
!
! CHECK-LABEL: --- ROUTINE test_array_sum (sptr# {{[0-9]+}}) ---
!
! 1. Find the one block marked "HEAD".
!
! CHECK: Block{{[ ]+[0-9]+}}, {{.*}}label: {{[ ]*}}[[LABEL:[0-9]+]], {{.*}}flags: {{[^:]+}} HEAD{{$}}
! CHECK-NOT: Block{{[ ]+[0-9]+}}, {{.*}}flags: {{[^:]+}} HEAD{{$}}
!
! 2. Find the ILI Area Dump
!
! CHECK: ILI Area Dump
!
! 3. Search for a jump to the head block.
!
! CHECK: {{.*}} {{[IK]}}CJMPZ {{.*}} gt {{.*}}[[LABEL]]~{{.*}}
! CHECK-NOT: {{[IK]}}CJMPZ
subroutine test_array_sum(ret, arr)
  integer :: arr(:)
  integer, INTENT(out) :: ret
  ret = sum(arr)
end subroutine

! CHECK-LABEL: --- ROUTINE test_array_assign (sptr# {{[0-9]+}}) ---
! CHECK: Block{{[ ]+[0-9]+}}, {{.*}}label: {{[ ]*}}[[LABEL:[0-9]+]], {{.*}}flags: {{[^:]+}} HEAD{{$}}
! CHECK-NOT: Block{{[ ]+[0-9]+}}, {{.*}}flags: {{[^:]+}} HEAD{{$}}
! CHECK: ILI Area Dump
! CHECK: {{.*}} {{[IK]}}CJMPZ {{.*}} gt {{.*}}[[LABEL]]~{{.*}}
! CHECK-NOT: {{[IK]}}CJMPZ
subroutine test_array_assign(arr)
  integer :: arr(:)
  arr = 3
end subroutine


! CHECK-LABEL: --- ROUTINE test_do_loop (sptr# {{[0-9]+}}) ---
! CHECK: Block{{[ ]+[0-9]+}}, {{.*}}label: {{[ ]*}}[[LABEL:[0-9]+]], {{.*}}flags: {{[^:]+}} HEAD{{$}}
! CHECK-NOT: Block{{[ ]+[0-9]+}}, {{.*}}flags: {{[^:]+}} HEAD{{$}}
! CHECK: ILI Area Dump
! CHECK: {{.*}} {{[IK]}}CJMPZ {{.*}} gt {{.*}}[[LABEL]]~{{.*}}
! CHECK-NOT: {{[IK]}}CJMPZ
subroutine test_do_loop(ret)
  integer, intent(out) :: ret
  ret = 0
  do i=1, 10
    ret = ret + i
  end do
end subroutine

! CHECK-LABEL: --- ROUTINE test_do_loop_variable_trip_count (sptr# {{[0-9]+}}) ---
! CHECK: Block{{[ ]+[0-9]+}}, {{.*}}label: {{[ ]*}}[[LABEL:[0-9]+]], {{.*}}flags: {{[^:]+}} HEAD{{$}}
! CHECK-NOT: Block{{[ ]+[0-9]+}}, {{.*}}flags: {{[^:]+}} HEAD{{$}}
! CHECK: ILI Area Dump
! CHECK: {{.*}} {{[IK]}}CJMPZ {{.*}} gt {{.*}}[[LABEL]]~{{.*}}
! CHECK-NOT: {{[IK]}}CJMPZ
subroutine test_do_loop_variable_trip_count(ret, n)
  integer, intent(out) :: ret
  integer, intent(in) :: n
  ret = 0
  do i=1, n
    ret = ret + i
  end do
end subroutine

! CHECK-LABEL: --- ROUTINE test_nested_do_loop (sptr# {{[0-9]+}}) ---
! CHECK: Block{{[ ]+[0-9]+}}, {{.*}}label: {{[ ]*}}[[LABEL1:[0-9]+]], {{.*}}flags: {{[^:]+}} HEAD{{$}}
! CHECK: Block{{[ ]+[0-9]+}}, {{.*}}label: {{[ ]*}}[[LABEL2:[0-9]+]], {{.*}}flags: {{[^:]+}} HEAD{{$}}
! CHECK-NOT: Block{{[ ]+[0-9]+}}, {{.*}}flags: {{[^:]+}} HEAD{{$}}
! CHECK: ILI Area Dump
! CHECK: {{.*}} {{[IK]}}CJMPZ {{.*}} gt {{.*}}[[LABEL2]]~{{.*}}
! CHECK: {{.*}} {{[IK]}}CJMPZ {{.*}} gt {{.*}}[[LABEL1]]~{{.*}}
! CHECK-NOT: {{[IK]}}CJMPZ
subroutine test_nested_do_loop(ret)
  integer, intent(out) :: ret
  ret = 0
  do i=1, 10
    do j=1, 10
      ret = ret + i + j
    end do
  end do
end subroutine

end program
