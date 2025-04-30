! RUN: %flang -O0 -c %s 2>&1 | FileCheck %s

program align
implicit none

    !DIR$ ALIGN alignment
    type T1
        integer(kind=2)     :: f1
        integer(kind=4)     :: f2
    end type T1
! CHECK: F90-W-0280-Syntax error in directive ALIGN: non-integer alignment

    !DIR$ ALIGN -3
    type T2
        integer(kind=2)     :: f1
        integer(kind=4)     :: f2
    end type T2
! CHECK: F90-W-0280-Syntax error in directive ALIGN: non-integer alignment

    !DIR$ ALIGN 0
    type T3
        integer(kind=2)     :: f1
        integer(kind=4)     :: f2
    end type T3
! CHECK: F90-W-0280-Syntax error in directive ALIGN: non-power-of-2 alignment

    !DIR$ ALIGN 3
    type T4
        integer(kind=2)     :: f1
        integer(kind=4)     :: f2
    end type T4
! CHECK: F90-W-0280-Syntax error in directive ALIGN: non-power-of-2 alignment


end program align
