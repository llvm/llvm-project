// UNSUPPORTED: -zos, target={{.*}}-aix{{.*}}
// RUN: rm -rf %t
// RUN: %clang_cc1 -emit-llvm -o - -fmodules-cache-path=%t -DA -fmodules -fimplicit-module-maps -F %S/Inputs/exportas-link %s | FileCheck --check-prefix=CHECK_A %s
// CHECK_A: !llvm.linker.options = !{![[MODULE:[0-9]+]]}
// CHECK_A: ![[MODULE]] = !{!"-framework", !"SomeKit"}
#ifdef A
@import SomeKitCore;
@import SomeKit;
#endif

// RUN: %clang_cc1 -emit-llvm -o - -fmodules-cache-path=%t -DB -fmodules -fimplicit-module-maps -F %S/Inputs/exportas-link %s | FileCheck --check-prefix=CHECK_B %s
// CHECK_B: !llvm.linker.options = !{![[MODULE:[0-9]+]]}
// CHECK_B: ![[MODULE]] = !{!"-framework", !"SomeKit"}
#ifdef B
@import SomeKit;
@import SomeKitCore;
#endif

// RUN: %clang_cc1 -emit-llvm -o - -fmodules-cache-path=%t -DC -fmodules -fimplicit-module-maps -F %S/Inputs/exportas-link %s | FileCheck --check-prefix=CHECK_C %s
// CHECK_C: !llvm.linker.options = !{![[MODULE:[0-9]+]]}
// CHECK_C: ![[MODULE]] = !{!"-framework", !"SomeKitCore"}
#ifdef C
@import SomeKitCore;
#endif

// RUN: %clang_cc1 -emit-llvm -o - -fmodules-cache-path=%t -DD -fmodules -fimplicit-module-maps -F %S/Inputs/exportas-link %s | FileCheck --check-prefix=CHECK_D %s
// CHECK_D: !llvm.linker.options = !{![[MODULE:[0-9]+]]}
// CHECK_D: ![[MODULE]] = !{!"-framework", !"SomeKit"}
#ifdef D
@import SomeKit;
#endif

// RUN: %clang_cc1 -emit-llvm -o - -fmodules-cache-path=%t -DE -fmodules -fimplicit-module-maps -F %S/Inputs/exportas-link %s | FileCheck --check-prefix=CHECK_E %s
// CHECK_E: !llvm.linker.options = !{![[MODULE:[0-9]+]]}
// CHECK_E: ![[MODULE]] = !{!"-framework", !"SomeKitCore"}
#ifdef E
@import OtherKit;
#endif

// RUN: %clang_cc1 -emit-llvm -o - -fmodules-cache-path=%t -DF -fmodules -fimplicit-module-maps -F %S/Inputs/exportas-link %s | FileCheck --check-prefix=CHECK_F %s
// CHECK_F: !llvm.linker.options = !{![[MODULE:[0-9]+]]}
// CHECK_F: ![[MODULE]] = !{!"-framework", !"SomeKit"}
#ifdef F
@import OtherKit;
#endif
