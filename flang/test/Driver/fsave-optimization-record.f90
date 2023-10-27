! Tests for the '-f[no-]save-optimization-record[=<format>]' flag.

! Test opt_record flags get generated for fc1
! RUN: %flang -### %s 2>&1 \
! RUN:     -foptimization-record-file=%t.opt.yaml \
! RUN:   | FileCheck --check-prefix=YAML %s

! RUN: %flang -### %s 2>&1 \
! RUN:     -fsave-optimization-record \
! RUN:   | FileCheck --check-prefix=YAML %s


! Test -foptimization-record-file produces YAML file with given content
! RUN: rm -f %t.opt.yaml
! RUN: %flang -foptimization-record-file=%t.opt.yaml -c %s
! RUN: cat %t.opt.yaml | FileCheck %s


! Test -fsave-optimization-record produces YAML file with given content
! RUN: rm -f %t.opt.yaml
! RUN: %flang -fsave-optimization-record -c -o %t.o %s
! RUN: cat %t.opt.yaml | FileCheck %s

! RUN: rm -f %t.opt.yaml
! RUN: %flang -fsave-optimization-record -S -o %t.s %s
! RUN: cat %t.opt.yaml | FileCheck %s


! Produces an empty file
! RUN: rm -f %t.opt.yaml
! RUN: %flang -fsave-optimization-record -S -emit-llvm -o %t.ll %s
! RUN: cat %t.opt.yaml


!Test unknown format produces error
! RUN: not %flang -fsave-optimization-record=hello %s 2>&1 \
! RUN:   | FileCheck --check-prefix=CHECK-FORMAT-ERROR %s


! YAML: "-opt-record-file" "{{.+}}.opt.yaml"
! YAML: "-opt-record-format" "yaml"

! CHECK: --- !Analysis
! CHECK: Pass:            prologepilog
! CHECK: Name:            StackSize
! CHECK: Function:        _QQmain
! CHECK: Pass:            asm-printer
! CHECK: Name:            InstructionMix
! CHECK: Name:            InstructionCount

! CHECK-FORMAT-ERROR: error: unknown remark serializer format: 'hello'

program forttest
    implicit none
    integer :: n

    n = 1 * 1

end program forttest
