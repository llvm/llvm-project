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
! RUN: %flang -### -flto=thin %s 2>&1 | FileCheck %s \
! RUN:        --check-prefixes=%if system-darwin || system-aix %{THIN-LTO-ALL%} \
! RUN:        %else %{THIN-LTO-ALL,THIN-LTO-LINKER-PLUGIN%}

! RUN: %flang -### -flto=thin --target=powerpc64-aix %s 2>&1 | FileCheck %s \
! RUN:        --check-prefix THIN-LTO-LINKER-AIX

! RUN: not %flang -### -S -flto=somelto %s 2>&1 | FileCheck %s --check-prefix=ERROR

! FC1 tests. Check that it does not crash.
! RUN: %flang_fc1 -S %s -flto -o /dev/null
! RUN: %flang_fc1 -S %s -flto=full -o /dev/null
! RUN: %flang_fc1 -S %s -flto=thin -o /dev/null

! NO-LTO: "-fc1"
! NO-LTO-NOT: flto

! FULL-LTO: "-fc1"
! FULL-LTO-SAME: "-flto=full"

! THIN-LTO-ALL: flang-new: warning: the option '-flto=thin' is a work in progress
! THIN-LTO-ALL: "-fc1"
! THIN-LTO-ALL-SAME: "-flto=thin"
! THIN-LTO-LINKER-PLUGIN: "-plugin-opt=thinlto"
! THIN-LTO-LINKER-AIX: "-bdbg:thinlto"

! ERROR: error: unsupported argument 'somelto' to option '-flto=
