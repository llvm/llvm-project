// Verify that we don't pass unwanted options to device-side compilation when
// clang-cl is used for CUDA compilation.
// Note: %s must be preceded by --, otherwise it may be interpreted as a
// command-line option, e.g. on Mac where %s is commonly under /Users.

// REQUIRES: clang-driver
// REQUIRES: x86-registered-target
// REQUIRES: nvptx-registered-target

// -stack-protector should not be passed to device-side CUDA compilation
// RUN: %clang_cl -### -nocudalib -nocudainc -- %s 2>&1 | FileCheck -check-prefix=GS-default %s
// GS-default: "-cc1" "-triple" "nvptx{{(64)?}}-nvidia-cuda"
// GS-default-NOT: "-stack-protector"
// GS-default: "-cc1" "-triple"
// GS-default: "-stack-protector" "2"

// -exceptions should be passed to device-side compilation.
// RUN: %clang_cl /c /GX -### -nocudalib -nocudainc -- %s 2>&1 | FileCheck -check-prefix=GX %s
// GX: "-cc1" "-triple" "nvptx{{(64)?}}-nvidia-cuda"
// GX-NOT: "-fcxx-exceptions"
// GX-NOT: "-fexceptions"
// GX: "-cc1" "-triple"
// GX: "-fcxx-exceptions" "-fexceptions"

// /Gd should not override default calling convention on device side.
// RUN: %clang_cl /c /Gd -### -nocudalib -nocudainc -- %s 2>&1 | FileCheck -check-prefix=Gd %s
// Gd: "-cc1" "-triple" "nvptx{{(64)?}}-nvidia-cuda"
// Gd-NOT: "-fcxx-exceptions"
// Gd-NOT: "-fdefault-calling-conv=cdecl"
// Gd: "-cc1" "-triple"
// Gd: "-fdefault-calling-conv=cdecl"
