// Test that the driver adds an arch-specific subdirectory in
// {RESOURCE_DIR}/lib/linux to the linker search path and to '-rpath' for native
// compilations.
//
// -rpath only gets added during native compilation.  To keep the test simple,
// just test for x86_64-linux native compilation.
// REQUIRES: x86_64-linux
//
// Add LIBPATH but no RPATH for -fsanitizer=address w/o -shared-libasan
// RUN: %clang %s -### 2>&1 -fsanitize=undefined \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_arch_subdir \
// RUN:   | FileCheck --check-prefixes=FILEPATH,LIBPATH,NO-RPATH %s
//
// Add LIBPATH, RPATH for -fsanitize=address -shared-libasan
// RUN: %clang %s -### 2>&1 -target x86_64-linux \
// RUN:     -fsanitize=address -shared-libasan \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_arch_subdir \
// RUN:   | FileCheck --check-prefixes=FILEPATH,LIBPATH,RPATH %s
//
// Add LIBPATH, RPATH with -fsanitize=address for Android
// RUN: %clang %s -### 2>&1 -target x86_64-linux-android -fsanitize=address \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_arch_subdir \
// RUN:   | FileCheck --check-prefixes=FILEPATH,LIBPATH,RPATH %s
//
// Add LIBPATH, RPATH for OpenMP
// RUN: %clang %s -### 2>&1 -fopenmp \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_arch_subdir \
// RUN:   | FileCheck --check-prefixes=FILEPATH,LIBPATH,RPATH %s
//
// Add LIBPATH but no RPATH for ubsan (or any other sanitizer)
// RUN: %clang %s -### 2>&1 -fsanitize=undefined \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_arch_subdir \
// RUN:   | FileCheck --check-prefixes=FILEPATH,LIBPATH,NO-RPATH %s
//
// Add LIBPATH but no RPATH if no sanitizer or runtime is specified
// RUN: %clang %s -### 2>&1 \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_arch_subdir \
// RUN:   | FileCheck --check-prefixes=FILEPATH,LIBPATH,NO-RPATH %s
//
// Do not add LIBPATH or RPATH if arch-specific subdir doesn't exist
// RUN: %clang %s -### 2>&1 \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:   | FileCheck --check-prefixes=FILEPATH,NO-LIBPATH,NO-RPATH %s
//
//
// FILEPATH: "-x" "c" "[[FILE_PATH:.*]]/{{.*}}.c"
// LIBPATH: -L[[FILE_PATH]]/Inputs/resource_dir_with_arch_subdir/lib/linux/x86_64
// RPATH: "-rpath" "[[FILE_PATH]]/Inputs/resource_dir_with_arch_subdir/lib/linux/x86_64"
// NO-LIBPATH-NOT: -L{{.*}}Inputs/resource_dir
// NO-RPATH-NOT: "-rpath" {{.*}}/Inputs/resource_dir
