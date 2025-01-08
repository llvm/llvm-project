// RUN: %clang -v --target=i686-pc-windows-gnu -### %s 2>&1 | FileCheck -check-prefix=CHECK_DEFAULT %s
// RUN: %clang -v --target=i686-pc-windows-gnu -lmsvcr120 -### %s 2>&1 | FileCheck -check-prefix=CHECK_MSVCR120 %s
// RUN: %clang -v --target=i686-pc-windows-gnu -lucrtbase -### %s 2>&1 | FileCheck -check-prefix=CHECK_UCRTBASE %s
// RUN: %clang -v --target=i686-pc-windows-gnu -lucrt -### %s 2>&1 | FileCheck -check-prefix=CHECK_UCRT %s
// RUN: %clang -v --target=i686-pc-windows-gnu -lcrtdll -### %s 2>&1 | FileCheck -check-prefix=CHECK_CRTDLL %s

// CHECK_DEFAULT: "-lmingwex" "-lmsvcrt" "-ladvapi32"
// CHECK_DEFAULT-SAME: "-lmsvcrt" "-lkernel32" "{{.*}}crtend.o"
// CHECK_MSVCR120: "-lmsvcr120"
// CHECK_MSVCR120-SAME: "-lmingwex" "-ladvapi32"
// CHECK_UCRTBASE: "-lucrtbase"
// CHECK_UCRTBASE-SAME: "-lmingwex" "-ladvapi32"
// CHECK_UCRT: "-lucrt"
// CHECK_UCRT-SAME: "-lmingwex" "-ladvapi32"
// CHECK_CRTDLL: "-lcrtdll"
// CHECK_CRTDLL-SAME: "-lmingwex" "-ladvapi32"
