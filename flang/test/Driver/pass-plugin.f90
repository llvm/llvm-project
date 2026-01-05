! Verify that the static and dynamically loaded pass plugins work as expected.
! (Plugins can be loaded dynamically or be statically linked into the binary.)
! This tests that the plugin is correctly added to and executed as part of the
! optimization pipeline.

! UNSUPPORTED: system-windows

! REQUIRES: plugins, shell, examples

! RUN: %flang -S %s %loadbye -mllvm -wave-goodbye -o /dev/null \
! RUN: 2>&1 | FileCheck %s

! RUN: %flang_fc1 -S %s %loadbye -mllvm -wave-goodbye -o /dev/null \
! RUN: 2>&1 | FileCheck %s


! CHECK: Bye: empty_

subroutine empty
end subroutine empty
