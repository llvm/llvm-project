!RUN: %flang -gdwarf-4 -S -emit-llvm %s -o - | FileCheck %s

!CHECK-LABEL: define void @callee_
!CHECK: call void @llvm.dbg.declare(metadata ptr %"array$p", metadata [[DLOC:![0-9]+]]
!CHECK-NEXT: call void @llvm.dbg.declare(metadata ptr %"array$p", metadata [[ASSOCIATED:![0-9]+]]
!CHECK-NEXT: call void @llvm.dbg.declare(metadata ptr %"array$sd", metadata [[ARRAY:![0-9]+]], metadata !DIExpression())
!CHECK: [[ARRAY]] = !DILocalVariable(name: "array"
!CHECK-SAME: arg: 2
!CHECK-SAME: type: [[TYPE:![0-9]+]]
!CHECK: [[TYPE]] = !DICompositeType(tag: DW_TAG_array_type,
!CHECK-SAME: dataLocation: [[DLOC]], associated: [[ASSOCIATED]]

subroutine callee (array)
  integer, pointer :: array(:, :)
  integer :: local = 4

  do i=LBOUND (array, 2), UBOUND (array, 2), 1
     do j=LBOUND (array, 1), UBOUND (array, 1), 1
        write(*, fmt="(i4)", advance="no") array (j, i)
     end do
     print *, ""
  end do

  local = local / 2
  print *, "local = ", local
end subroutine callee

program caller

  interface
     subroutine callee (array)
       integer, pointer :: array(:, :)
     end subroutine callee
  end interface

  integer, pointer :: caller_arr(:, :)
  integer, target :: tgt_arr(10,10)
  tgt_arr = 99
  tgt_arr(2,2) = 88

  caller_arr => tgt_arr
  call callee (caller_arr)

  print *, ""
end program caller
