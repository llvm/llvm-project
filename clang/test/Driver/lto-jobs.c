// Confirm that -flto-jobs=N is passed to linker

// RUN: %clang --target=x86_64-unknown-linux -### %s -flto=thin -flto-jobs=5 2> %t
// RUN: FileCheck -check-prefix=CHECK-LINK-THIN-JOBS-ACTION < %t %s
//
// RUN: %clang --target=x86_64-sie-ps5 -### %s -flto=thin -flto-jobs=5 2> %t
// RUN: FileCheck -check-prefix=CHECK-LINK-THIN-JOBS-ACTION < %t %s
//
// RUN: %clang --target=x86_64-sie-ps5 -### %s -flto-jobs=5 2> %t
// RUN: FileCheck -check-prefix=CHECK-LINK-THIN-JOBS-ACTION < %t %s
//
// CHECK-LINK-THIN-JOBS-ACTION: "-plugin-opt=jobs=5"
//
// RUN: %clang --target=x86_64-scei-ps4 -### %s -flto=thin -flto-jobs=5 2> %t
// RUN: FileCheck -check-prefix=CHECK-PS4-LINK-THIN-JOBS-ACTION < %t %s
//
// CHECK-PS4-LINK-THIN-JOBS-ACTION: "-lto-debug-options= -threads=5"

// RUN: %clang --target=x86_64-apple-darwin13.3.0 -### %s -flto=thin -flto-jobs=5 2> %t
// RUN: FileCheck -check-prefix=CHECK-LINK-THIN-JOBS2-ACTION < %t %s
//
// CHECK-LINK-THIN-JOBS2-ACTION: "-mllvm" "-threads={{[0-9]+}}"

// RUN: %clang --target=powerpc-ibm-aix -### %s -flto=thin -flto-jobs=5 2> %t
// RUN: FileCheck -check-prefix=CHECK-AIX-LINK-THIN-JOBS-ACTION < %t %s
//
// CHECK-AIX-LINK-THIN-JOBS-ACTION: "-bplugin_opt:-threads=5"
