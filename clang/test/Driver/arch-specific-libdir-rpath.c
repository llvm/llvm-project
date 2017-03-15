// Test that the driver adds an arch-specific subdirectory in
// {RESOURCE_DIR}/lib/linux to the linker search path and to '-rpath'
//
// Test the default behavior when neither -frtlib-add-rpath nor
// -fno-rtlib-add-rpath is specified, which is to skip -rpath
// RUN: %clang %s -### 2>&1 -target x86_64-linux \
// RUN:     -fsanitize=address -shared-libasan \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_arch_subdir \
// RUN:     -frtlib-add-rpath \
// RUN:   | FileCheck --check-prefixes=RESDIR,LIBPATH-X86_64,NO-RPATH-X86_64 %s
//
// Test that -rpath is not added under -fno-rtlib-add-rpath even if other
// conditions are met.
// RUN: %clang %s -### 2>&1 -target x86_64-linux \
// RUN:     -fsanitize=address -shared-libasan \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_arch_subdir \
// RUN:     -frtlib-add-rpath \
// RUN:   | FileCheck --check-prefixes=RESDIR,LIBPATH-X86_64,NO-RPATH-X86_64 %s
//
// Test that -rpath is added only under the right circumstance even if
// -frtlib-add-rpath is specified.
//
// Add LIBPATH but no RPATH for -fsanitizer=address w/o -shared-libasan
// RUN: %clang %s -### 2>&1 -target x86_64-linux -fsanitize=undefined \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_arch_subdir \
// RUN:     -frtlib-add-rpath \
// RUN:   | FileCheck --check-prefixes=RESDIR,LIBPATH-X86_64,NO-RPATH %s
//
// Add LIBPATH but no RPATH for -fsanitizer=address w/o -shared-libasan
// RUN: %clang %s -### 2>&1 -target x86_64-linux -fsanitize=undefined \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_arch_subdir \
// RUN:     -frtlib-add-rpath \
// RUN:   | FileCheck --check-prefixes=RESDIR,LIBPATH-X86_64,NO-RPATH %s
//
// Add LIBPATH, RPATH for -fsanitize=address -shared-libasan
// RUN: %clang %s -### 2>&1 -target x86_64-linux \
// RUN:     -fsanitize=address -shared-libasan \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_arch_subdir \
// RUN:     -frtlib-add-rpath \
// RUN:   | FileCheck --check-prefixes=RESDIR,LIBPATH-X86_64,RPATH-X86_64 %s
//
// Add LIBPATH, RPATH for -fsanitize=address -shared-libasan on aarch64
// RUN: %clang %s -### 2>&1 -target aarch64-linux \
// RUN:     -fsanitize=address -shared-libasan \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_arch_subdir \
// RUN:     -frtlib-add-rpath \
// RUN:   | FileCheck --check-prefixes=RESDIR,LIBPATH-AArch64,RPATH-AArch64 %s
//
// Add LIBPATH, RPATH with -fsanitize=address for Android
// RUN: %clang %s -### 2>&1 -target x86_64-linux-android -fsanitize=address \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_arch_subdir \
// RUN:     -frtlib-add-rpath \
// RUN:   | FileCheck --check-prefixes=RESDIR,LIBPATH-X86_64,RPATH-X86_64 %s
//
// Add LIBPATH, RPATH for OpenMP
// RUN: %clang %s -### 2>&1 -target x86_64-linux -fopenmp \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_arch_subdir \
// RUN:     -frtlib-add-rpath \
// RUN:   | FileCheck --check-prefixes=RESDIR,LIBPATH-X86_64,RPATH-X86_64 %s
//
// Add LIBPATH but no RPATH for ubsan (or any other sanitizer)
// RUN: %clang %s -### 2>&1 -fsanitize=undefined -target x86_64-linux \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_arch_subdir \
// RUN:     -frtlib-add-rpath \
// RUN:   | FileCheck --check-prefixes=RESDIR,LIBPATH-X86_64,NO-RPATH %s
//
// Add LIBPATH but no RPATH if no sanitizer or runtime is specified
// RUN: %clang %s -### 2>&1 -target x86_64-linux \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_arch_subdir \
// RUN:     -frtlib-add-rpath \
// RUN:   | FileCheck --check-prefixes=RESDIR,LIBPATH-X86_64,NO-RPATH %s
//
// Do not add LIBPATH or RPATH if arch-specific subdir doesn't exist
// RUN: %clang %s -### 2>&1 -target x86_64-linux \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     -frtlib-add-rpath \
// RUN:   | FileCheck --check-prefixes=RESDIR,NO-LIBPATH,NO-RPATH %s
//
// RESDIR: "-resource-dir" "[[RESDIR:[^"]*]]"
// LIBPATH-X86_64:  -L[[RESDIR]]{{(/|\\\\)lib(/|\\\\)linux(/|\\\\)x86_64}}
// RPATH-X86_64:    "-rpath" "[[RESDIR]]{{(/|\\\\)lib(/|\\\\)linux(/|\\\\)x86_64}}"
// LIBPATH-AArch64: -L[[RESDIR]]{{(/|\\\\)lib(/|\\\\)linux(/|\\\\)aarch64}}
// RPATH-AArch64:   "-rpath" "[[RESDIR]]{{(/|\\\\)lib(/|\\\\)linux(/|\\\\)aarch64}}"
// NO-LIBPATH-NOT:  -L{{.*Inputs(/|\\\\)resource_dir}}
// NO-RPATH-NOT:    "-rpath" {{.*(/|\\\\)Inputs(/|\\\\)resource_dir}}
