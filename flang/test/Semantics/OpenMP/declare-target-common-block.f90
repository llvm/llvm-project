! RUN: %flang_fc1 -fopenmp -fdebug-dump-symbols %s | FileCheck %s

PROGRAM main
    !CHECK: one (OmpDeclareTarget) size=4 offset=0: ObjectEntity type: REAL(4)
    !CHECK: two (OmpDeclareTarget) size=4 offset=4: ObjectEntity type: REAL(4)
    !CHECK: numbers size=8 offset=0: CommonBlockDetails alignment=4: one two
    REAL :: one, two
    COMMON /numbers/ one, two
    !$omp declare target(/numbers/)
END
