// UNSUPPORTED: system-windows

// Test that /lib64 is used if /lib doesn't exist.
// RUN: rm -rf %t/*
// RUN: mkdir -p %t/rocm
// RUN: cp -R %S/Inputs/rocm/bin %t/rocm/bin
// RUN: cp -R %S/Inputs/rocm/amdgcn %t/rocm/amdgcn
// RUN: cp -R %S/Inputs/rocm/include %t/rocm/include
// RUN: cp -R %S/Inputs/rocm/lib %t/rocm/lib64

// RUN: %clang -### -target x86_64-linux-gnu --offload-arch=gfx1010 \
// RUN:   --rocm-path=%t/rocm --hip-link %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=LIB64 %s

// RUN: %clang -### -target x86_64-windows-msvc --offload-arch=gfx1010 \
// RUN:   --rocm-path=%t/rocm --hip-link %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=WIN_LIB64 %s

// Test that in the presence of both /lib and /lib64, the former is preferred.
// RUN: cp -R %S/Inputs/rocm/lib %t/rocm/lib

// RUN: %clang -### -target x86_64-linux-gnu --offload-arch=gfx1010 \
// RUN:   --rocm-path=%t/rocm --hip-link %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=LIB %s

// RUN: %clang -### -target x86_64-windows-msvc --offload-arch=gfx1010 \
// RUN:   --rocm-path=%t/rocm --hip-link %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=WIN_LIB %s

// LIB64: "[[ROCM:[^"]+/rocm]]/lib64/libamdhip64.so"
// LIB64: "-L[[ROCM]]/lib64"
// LIB64-NOT: "[[ROCM]]/lib/libamdhip64.so"

// LIB: "[[ROCM:[^"]+/rocm]]/lib/libamdhip64.so"
// LIB: "-L[[ROCM]]/lib"
// LIB-NOT: "[[ROCM]]/lib64/libamdhip64.so"

// WIN_LIB64: "-libpath:{{.*rocm[/\\]lib64}}" "amdhip64.lib"
// WIN_LIB64-NOT: "-libpath:{{.*rocm[/\\]lib}}" "amdhip64.lib"

// WIN_LIB: "-libpath:{{.*rocm[/\\]lib}}" "amdhip64.lib"
// WIN_LIB-NOT: "-libpath:{{.*rocm[/\\]lib64}}" "amdhip64.lib"
