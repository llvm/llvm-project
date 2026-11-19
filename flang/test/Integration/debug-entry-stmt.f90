! Checks that each ENTRY statement gets its own DISubprogram, with line and
! scopeLine set to the ENTRY statement's own line, with and without dummy
! arguments or local variables.

! RUN: %flang_fc1 -emit-llvm -debug-info-kind=standalone -O0 %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-llvm -debug-info-kind=standalone -O2 %s -o - | FileCheck %s

subroutine sub_with_arg(i)
    implicit none
    integer :: i, result
    result = i + 1
    goto 100
entry entry_with_arg(i)
    result = i + 100
100 print *, result
end subroutine sub_with_arg

! CHECK: distinct !DISubprogram(name: "entry_with_arg",
! CHECK-SAME: line: 13, {{.*}}scopeLine: 13

! An entry with no dummy arguments of its own.
subroutine sub_no_arg(i)
    implicit none
    integer :: i, result
    result = i + 1
    goto 200
entry entry_no_arg()
    result = 42
200 print *, result
end subroutine sub_no_arg

! CHECK: distinct !DISubprogram(name: "entry_no_arg",
! CHECK-SAME: line: 27, {{.*}}scopeLine: 27

! An entry with neither dummy arguments nor local variables.
subroutine sub_bare()
    goto 300
entry entry_bare()
    print *, 0
300 continue
end subroutine sub_bare

! CHECK: distinct !DISubprogram(name: "entry_bare",
! CHECK-SAME: line: 38, {{.*}}scopeLine: 38
