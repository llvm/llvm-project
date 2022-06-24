// Test that the driver adds an arch-specific subdirectory in
// {RESOURCE_DIR}/lib/linux to the search path.
//
// RUN: %clang %s -### 2>&1 -target i386-unknown-linux \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_arch_subdir \
// RUN:   | FileCheck --check-prefixes=FILEPATH,ARCHDIR-i386 %s
//
// RUN: %clang %s -### 2>&1 -target i386-unknown-linux \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:   | FileCheck --check-prefixes=FILEPATH,NO-ARCHDIR %s
//
// RUN: %clang %s -### 2>&1 -target i686-unknown-linux \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_arch_subdir \
// RUN:   | FileCheck --check-prefixes=FILEPATH,ARCHDIR-i386 %s
//
// RUN: %clang %s -### 2>&1 -target i686-unknown-linux \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:   | FileCheck --check-prefixes=FILEPATH,NO-ARCHDIR %s
//
// RUN: %clang %s -### 2>&1 -target x86_64-unknown-linux \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_arch_subdir \
// RUN:   | FileCheck --check-prefixes=FILEPATH,ARCHDIR-x86_64 %s
//
// RUN: %clang %s -### 2>&1 -target x86_64-unknown-linux \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:   | FileCheck --check-prefixes=FILEPATH,NO-ARCHDIR %s
//
// RUN: %clang %s -### 2>&1 -target arm-unknown-linux \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_arch_subdir \
// RUN:   | FileCheck --check-prefixes=FILEPATH,ARCHDIR-arm %s
//
// RUN: %clang %s -### 2>&1 -target arm-unknown-linux \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:   | FileCheck --check-prefixes=FILEPATH,NO-ARCHDIR %s
//
// RUN: %clang %s -### 2>&1 -target aarch64-unknown-linux \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_arch_subdir \
// RUN:   | FileCheck --check-prefixes=FILEPATH,ARCHDIR-aarch64 %s
//
// RUN: %clang %s -### 2>&1 -target aarch64-unknown-linux \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:   | FileCheck --check-prefixes=FILEPATH,NO-ARCHDIR %s
//
//
// FILEPATH: "-x" "c" "[[FILE_PATH:.*]]{{(/|\\\\).*}}.c"
// ARCHDIR-i386:    -L[[FILE_PATH]]{{(/|\\\\)Inputs(/|\\\\)resource_dir_with_arch_subdir(/|\\\\)lib(/|\\\\)linux(/|\\\\)i386}}
// ARCHDIR-x86_64:  -L[[FILE_PATH]]{{(/|\\\\)Inputs(/|\\\\)resource_dir_with_arch_subdir(/|\\\\)lib(/|\\\\)linux(/|\\\\)x86_64}}
// ARCHDIR-arm:     -L[[FILE_PATH]]{{(/|\\\\)Inputs(/|\\\\)resource_dir_with_arch_subdir(/|\\\\)lib(/|\\\\)linux(/|\\\\)arm}}
// ARCHDIR-aarch64: -L[[FILE_PATH]]{{(/|\\\\)Inputs(/|\\\\)resource_dir_with_arch_subdir(/|\\\\)lib(/|\\\\)linux(/|\\\\)aarch64}}
//
// Have a stricter check for no-archdir - that the driver doesn't add any
// subdirectory from the provided resource directory.
// NO-ARCHDIR-NOT: -L[[FILE_PATH]]/Inputs/resource_dir
