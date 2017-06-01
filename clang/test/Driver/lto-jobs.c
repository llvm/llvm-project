// Confirm that -flto-jobs=N is passed to linker

// RUN: %clang -target x86_64-unknown-linux -### %s -flto=thin -flto-jobs=5 2> %t
// RUN: FileCheck -check-prefix=CHECK-LINK-THIN-JOBS-ACTION < %t %s
//
// CHECK-LINK-THIN-JOBS-ACTION: "-plugin-opt=jobs=5"

// RUN: %clang -target x86_64-apple-darwin13.3.0 -### %s -flto=thin -flto-jobs=5 2> %t
// RUN: FileCheck -check-prefix=CHECK-LINK-THIN-JOBS2-ACTION < %t %s
//
// CHECK-LINK-THIN-JOBS2-ACTION: "-mllvm" "-threads=5"
