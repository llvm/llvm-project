! Test that -gline-directives-only leaves the name table off.

! RUN: %flang_fc1 -debug-info-kind=line-directives-only -emit-llvm -o - %s \
! RUN:   | FileCheck %s

! CHECK: !DICompileUnit({{.*}}emissionKind: DebugDirectivesOnly
! CHECK-SAME: nameTableKind: None

program test
end program test
