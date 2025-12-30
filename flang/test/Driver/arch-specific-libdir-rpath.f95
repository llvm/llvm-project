! REQUIRES: x86-registered-target
! Test that the driver adds an arch-specific subdirectory in
! {RESOURCE_DIR}/lib/linux to the linker search path and to '-rpath'
!
! Test the default behavior when neither -frtlib-add-rpath nor
! -fno-rtlib-add-rpath is specified, which is to skip -rpath
! RUN: %flang %s -### --target=x86_64-linux \
! RUN:     -resource-dir=%S/../../../clang/test/Driver/Inputs/resource_dir_with_arch_subdir 2>&1 \
! RUN:   | FileCheck --check-prefixes=RESDIR,LIBPATH-X86_64,NO-RPATH-X86_64 %s
!
! Test that -rpath is not added under -fno-rtlib-add-rpath
! RUN: %flang %s -### --target=x86_64-linux \
! RUN:     -resource-dir=%S/../../../clang/test/Driver/Inputs/resource_dir_with_arch_subdir \
! RUN:     -fno-rtlib-add-rpath 2>&1 \
! RUN:   | FileCheck --check-prefixes=RESDIR,LIBPATH-X86_64,NO-RPATH-X86_64 %s
!
! Test that -rpath is added
!
! Add LIBPATH, RPATH
!
! RUN: %flang %s -### --target=x86_64-linux \
! RUN:     -resource-dir=%S/../../../clang/test/Driver/Inputs/resource_dir_with_arch_subdir \
! RUN:     -frtlib-add-rpath 2>&1 \
! RUN:   | FileCheck --check-prefixes=RESDIR,LIBPATH-X86_64,RPATH-X86_64 %s
!
! Add LIBPATH, RPATH for OpenMP
!
! RUN: %flang %s -### --target=x86_64-linux -fopenmp \
! RUN:     -resource-dir=%S/../../../clang/test/Driver/Inputs/resource_dir_with_arch_subdir \
! RUN:     -frtlib-add-rpath 2>&1 \
! RUN:   | FileCheck --check-prefixes=RESDIR,LIBPATH-X86_64,RPATH-X86_64 %s
!
!
! RESDIR: "-resource-dir" "[[RESDIR:[^"]*]]"
!
! LIBPATH-X86_64: -L[[RESDIR]]{{(/|\\\\)lib(/|\\\\)linux(/|\\\\)x86_64}}
! RPATH-X86_64:   "-rpath" "[[RESDIR]]{{(/|\\\\)lib(/|\\\\)linux(/|\\\\)x86_64}}"
!
! NO-RPATH-X86_64-NOT:   "-rpath" "[[RESDIR]]{{(/|\\\\)lib(/|\\\\)linux(/|\\\\)x86_64}}"
