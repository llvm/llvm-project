! Test that the compile unit asks for a name table only when -gpubnames or
! -ggnu-pubnames is given, and that Apple targets always ask for Apple ones.

! REQUIRES: x86-registered-target

! RUN: %flang_fc1 -triple x86_64-unknown-linux-gnu -emit-llvm \
! RUN:   -debug-info-kind=standalone -o - %s | FileCheck %s --check-prefix=NONE
! RUN: %flang_fc1 -triple x86_64-unknown-linux-gnu -emit-llvm \
! RUN:   -debug-info-kind=standalone -gpubnames -o - %s \
! RUN:   | FileCheck %s --check-prefix=DEFAULT
! RUN: %flang_fc1 -triple x86_64-unknown-linux-gnu -emit-llvm \
! RUN:   -debug-info-kind=standalone -ggnu-pubnames -o - %s \
! RUN:   | FileCheck %s --check-prefix=GNU
! RUN: %flang_fc1 -triple x86_64-apple-macosx -emit-llvm \
! RUN:   -debug-info-kind=standalone -o - %s | FileCheck %s --check-prefix=APPLE
! RUN: %flang_fc1 -triple x86_64-apple-macosx -emit-llvm \
! RUN:   -debug-info-kind=standalone -ggnu-pubnames -o - %s \
! RUN:   | FileCheck %s --check-prefix=APPLE

! NONE: !DICompileUnit({{.*}}nameTableKind: None
! DEFAULT: !DICompileUnit(
! DEFAULT-NOT: nameTableKind
! GNU: !DICompileUnit({{.*}}nameTableKind: GNU
! APPLE: !DICompileUnit({{.*}}nameTableKind: Apple

program test
end program test
