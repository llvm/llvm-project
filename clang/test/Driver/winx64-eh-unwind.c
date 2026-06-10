// RUN: %clang -### --target=x86_64-windows-msvc -fwinx64-eh-unwind=v1 %s 2>&1 | FileCheck --check-prefix=V1 %s
// RUN: %clang -### --target=x86_64-windows-msvc -fwinx64-eh-unwind=v2-best-effort %s 2>&1 | FileCheck --check-prefix=V2BE %s
// RUN: %clang -### --target=x86_64-windows-msvc -fwinx64-eh-unwind=v2-required %s 2>&1 | FileCheck --check-prefix=V2REQ %s
// RUN: %clang -### --target=x86_64-windows-msvc -fwinx64-eh-unwind=v3 %s 2>&1 | FileCheck --check-prefix=V3 %s

// Legacy -fwinx64-eh-unwindv2= translation.
// RUN: %clang -### --target=x86_64-windows-msvc -fwinx64-eh-unwindv2=best-effort %s 2>&1 | FileCheck --check-prefix=V2BE %s
// RUN: %clang -### --target=x86_64-windows-msvc -fwinx64-eh-unwindv2=required %s 2>&1 | FileCheck --check-prefix=V2REQ %s
// Legacy disabled maps to v1 default — no flag should be forwarded.
// RUN: %clang -### --target=x86_64-windows-msvc -fwinx64-eh-unwindv2=disabled %s 2>&1 | FileCheck --check-prefix=V1DISABLED %s

// MSVC compatibility flags.
// RUN: %clang_cl -### --target=x86_64-windows-msvc /d2epilogunwind -- %s 2>&1 | FileCheck --check-prefix=V2BE %s
// RUN: %clang_cl -### --target=x86_64-windows-msvc /d2epilogunwindrequirev2 -- %s 2>&1 | FileCheck --check-prefix=V2REQ %s

// V1:          "-fwinx64-eh-unwind=v1"
// V1DISABLED-NOT: "-fwinx64-eh-unwind=
// V2BE:        "-fwinx64-eh-unwind=v2-best-effort"
// V2REQ:       "-fwinx64-eh-unwind=v2-required"
// V3:          "-fwinx64-eh-unwind=v3"

void f(void) {}
