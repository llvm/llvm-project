! RUN: %flang -O2 -mllvm -force-specialization -Xflang -fdebug-pass-manager -S -emit-llvm %s -o %t.O2.ll 2>%t.O2.stderr
! RUN: FileCheck %s --check-prefix=PASS-O2-PIPE < %t.O2.stderr
! RUN: FileCheck %s --check-prefix=PASS-O2-IR < %t.O2.ll
!
! RUN: %flang -O3 -mllvm -force-specialization -Xflang -fdebug-pass-manager -S -emit-llvm %s -o %t.O3.ll 2>%t.O3.stderr
! RUN: FileCheck %s --check-prefix=PASS-O3-PIPE < %t.O3.stderr
! RUN: FileCheck %s --check-prefix=PASS-O3-IR < %t.O3.ll

module brute_force
  implicit none
  integer, public :: fallback_sink
  integer, volatile, public :: global_fence = 1
contains
  subroutine top_level_caller()
    integer :: temp
    temp = 2
    call digits_2(temp)
  end subroutine top_level_caller

  recursive subroutine digits_2(arg1)
    integer, intent(in) :: arg1
    integer :: temp_inner

    if (global_fence == 1) then
      if (arg1 == 2) then
        temp_inner = arg1
        call digits_2(temp_inner)
      else
        fallback_sink = arg1 * 5 + 12
      end if
    end if
  end subroutine digits_2
end module brute_force

! PASS-O2-PIPE:      Running pass: IPSCCPPass on [module]
! PASS-O2-PIPE:      Running pass: InlinerPass on (
! PASS-O2-PIPE-NOT:  Running pass: IPSCCPPass on [module]
! PASS-O2-PIPE:      Running pass: DeadArgumentEliminationPass on [module]

! PASS-O2-IR-NOT:    .specialized.

! PASS-O3-PIPE:      Running pass: IPSCCPPass on [module]
! PASS-O3-PIPE:      Running pass: InlinerPass on (
! PASS-O3-PIPE:      Running pass: IPSCCPPass on [module]
! PASS-O3-PIPE:      Running pass: DeadArgumentEliminationPass on [module]

! PASS-O3-IR:        define void @{{.*}}top_level_caller
! PASS-O3-IR:        call fastcc void @{{.*}}digits_2{{.*}}.specialized.{{.*}}()
! PASS-O3-IR:        define internal fastcc void @{{.*}}digits_2{{.*}}.specialized.1()
! PASS-O3-IR:        define internal fastcc void @{{.*}}digits_2{{.*}}.specialized.2()
