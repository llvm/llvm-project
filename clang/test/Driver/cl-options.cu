// Verify that we don't pass unwanted options to device-side compilation when
// clang-cl is used for CUDA compilation.
// Note: %s must be preceded by --, otherwise it may be interpreted as a
// command-line option, e.g. on Mac where %s is commonly under /Users.

// -stack-protector should not be passed to device-side CUDA compilation
// RUN: not %clang_cl -### -nocudalib -nocudainc -- %s 2>&1 | FileCheck -check-prefix=GS-default %s
// GS-default: "-cc1" "-triple" "nvptx{{(64)?}}-nvidia-cuda"
// GS-default-NOT: "-stack-protector"
// GS-default: "-cc1" "-triple"
// GS-default: "-stack-protector" "2"

// -exceptions should be passed to device-side compilation.
// RUN: not %clang_cl /c /GX -### -nocudalib -nocudainc -- %s 2>&1 | FileCheck -check-prefix=GX %s
// GX: "-cc1" "-triple" "nvptx{{(64)?}}-nvidia-cuda"
// GX-NOT: "-fcxx-exceptions"
// GX-NOT: "-fexceptions"
// GX: "-cc1" "-triple"
// GX: "-fcxx-exceptions" "-fexceptions"

// /Gd should not override default calling convention on device side.
// RUN: not %clang_cl /c /Gd -### -nocudalib -nocudainc -- %s 2>&1 | FileCheck -check-prefix=Gd %s
// Gd: "-cc1" "-triple" "nvptx{{(64)?}}-nvidia-cuda"
// Gd-NOT: "-fcxx-exceptions"
// Gd-NOT: "-fdefault-calling-conv=cdecl"
// Gd: "-cc1" "-triple"
// Gd: "-fdefault-calling-conv=cdecl"

// Optimization options must continue to apply to both CUDA compilation jobs.
// RUN: not %clang_cl /c /O2 -### -nocudalib -nocudainc -- %s 2>&1 | FileCheck -check-prefix=O2 %s
// O2: "-cc1" "-triple" "nvptx{{(64)?}}-nvidia-cuda"
// O2-SAME: "-O3"
// O2: "-cc1" "-triple"
// O2-SAME: "-O3"

// Shared macros and language settings also reach CUDA without reintroducing
// the original macro alongside its translated value.
// RUN: not %clang_cl --target=x86_64-pc-windows-msvc -### /c \
// RUN:   --cuda-gpu-arch=sm_35 -nocudainc -nocudalib \
// RUN:   /DFOO#7 /permissive- -- %s 2>&1 | FileCheck %s --check-prefix=CL-COMMON
// CL-COMMON: "-cc1" "-triple" "nvptx64-nvidia-cuda"
// CL-COMMON-SAME: "-D" "FOO=7"
// CL-COMMON-NOT: "FOO#7"
// CL-COMMON-NOT: "-fno-operator-names"
// CL-COMMON-NOT: "-fdelayed-template-parsing"
// CL-COMMON: "-cc1" "-triple" "x86_64-pc-windows-msvc{{[^"]*}}"
// CL-COMMON-SAME: "-D" "FOO=7"
// CL-COMMON-NOT: "FOO#7"
// CL-COMMON-NOT: "-fno-operator-names"
// CL-COMMON-NOT: "-fdelayed-template-parsing"
