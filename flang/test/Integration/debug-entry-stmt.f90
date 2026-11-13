! Checks that each ENTRY statement gets its own DISubprogram with a
! source location and prologue_end marker at its first executable
! statement, across optimization levels and with/without dummy
! arguments or local variables.

! RUN: %flang_fc1 -emit-llvm -debug-info-kind=standalone -O0 %s -o - | FileCheck %s --check-prefixes=IR,IR-O0
! RUN: %flang_fc1 -emit-llvm -debug-info-kind=standalone -O1 %s -o - | FileCheck %s --check-prefix=IR
! RUN: %flang_fc1 -emit-llvm -debug-info-kind=standalone -O2 %s -o - | FileCheck %s --check-prefix=IR
! RUN: %flang_fc1 -emit-llvm -debug-info-kind=standalone -O3 %s -o - | FileCheck %s --check-prefix=IR
! RUN: %if x86-registered-target %{ \
! RUN:   %flang_fc1 -S -debug-info-kind=standalone -triple x86_64-unknown-linux-gnu -O0 %s -o - \
! RUN:     | FileCheck %s --check-prefixes=ASM,ASM-O0 && \
! RUN:   %flang_fc1 -S -debug-info-kind=standalone -triple x86_64-unknown-linux-gnu -O1 %s -o - \
! RUN:     | FileCheck %s --check-prefix=ASM && \
! RUN:   %flang_fc1 -S -debug-info-kind=standalone -triple x86_64-unknown-linux-gnu -O2 %s -o - \
! RUN:     | FileCheck %s --check-prefix=ASM && \
! RUN:   %flang_fc1 -S -debug-info-kind=standalone -triple x86_64-unknown-linux-gnu -O3 %s -o - \
! RUN:     | FileCheck %s --check-prefix=ASM \
! RUN: %}


subroutine sub_with_arg(i)
    implicit none
    integer :: i, result
    result = i + 1
    goto 100
entry entry_with_arg(i)
    result = i + 100
100 print *, result
end subroutine sub_with_arg

! IR: ![[SUB_WITH_ARG:[0-9]+]] = distinct !DISubprogram(name: "sub_with_arg"
! IR: ![[ENTRY_WITH_ARG:[0-9]+]] = distinct !DISubprogram(name: "entry_with_arg",
! IR-SAME: line: 27, {{.*}}scopeLine: 27
! IR: !DILocation(line: 28, {{.*}}scope: ![[ENTRY_WITH_ARG]])
!
! ASM-LABEL: sub_with_arg_:
! ASM-LABEL: entry_with_arg_:
! ASM: .loc 1 28 {{.*}} prologue_end

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

! IR: ![[SUB_NO_ARG:[0-9]+]] = distinct !DISubprogram(name: "sub_no_arg"
! IR: ![[ENTRY_NO_ARG:[0-9]+]] = distinct !DISubprogram(name: "entry_no_arg",
! IR-SAME: line: 47, {{.*}}scopeLine: 47
! IR-O0: !DILocation(line: 48, {{.*}}scope: ![[ENTRY_NO_ARG]])
!
! ASM-LABEL: sub_no_arg_:
! ASM-LABEL: entry_no_arg_:
! ASM-O0: .loc 1 48 {{.*}} prologue_end

! An entry with neither dummy arguments nor local variables.
subroutine sub_bare()
    goto 300
entry entry_bare()
    print *, 0
300 continue
end subroutine sub_bare

! IR: ![[SUB_BARE:[0-9]+]] = distinct !DISubprogram(name: "sub_bare"
! IR: ![[ENTRY_BARE:[0-9]+]] = distinct !DISubprogram(name: "entry_bare",
! IR-SAME: line: 64, {{.*}}scopeLine: 64
! IR: !DILocation(line: 65, {{.*}}scope: ![[ENTRY_BARE]])
!
! ASM-LABEL: entry_bare_:
! ASM: .loc 1 65 {{.*}} prologue_end
