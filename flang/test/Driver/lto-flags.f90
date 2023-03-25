! Solaris ld doesn't support the linker plugin interface
! UNSUPPORTED: system-windows, system-solaris
! RUN: %flang -### -S %s 2>&1 | FileCheck %s --check-prefix=NO-LTO
! RUN: %flang -### -S -fno-lto %s 2>&1 | FileCheck %s --check-prefix=NO-LTO

! Full LTO and aliases.
! RUN: %flang -### -S -flto %s 2>&1 | FileCheck %s --check-prefix=FULL-LTO
! RUN: %flang -### -S -flto=full %s 2>&1 | FileCheck %s --check-prefix=FULL-LTO
! RUN: %flang -### -S -flto=auto %s 2>&1 | FileCheck %s --check-prefix=FULL-LTO
! RUN: %flang -### -S -flto=jobserver %s 2>&1 | FileCheck %s --check-prefix=FULL-LTO

! Also check linker plugin opt for Thin LTO
! RUN: %flang -### -flto=thin %s 2>&1 | FileCheck %s --check-prefix=THIN-LTO

! RUN: %flang -### -S -flto=somelto %s 2>&1 | FileCheck %s --check-prefix=ERROR

! FC1 tests. Check that it does not crash.
! RUN: %flang_fc1 -S %s -flto -o /dev/null
! RUN: %flang_fc1 -S %s -flto=full -o /dev/null
! RUN: %flang_fc1 -S %s -flto=thin -o /dev/null

! NO-LTO: "-fc1"
! NO-LTO-NOT: flto

! FULL-LTO: "-fc1"
! FULL-LTO-SAME: "-flto=full"

! THIN-LTO: flang-new: warning: the option '-flto=thin' is a work in progress
! THIN-LTO: "-fc1"
! THIN-LTO-SAME: "-flto=thin"
! THIN-LTO: "-plugin-opt=thinlto"

! ERROR: error: unsupported argument 'somelto' to option '-flto=
