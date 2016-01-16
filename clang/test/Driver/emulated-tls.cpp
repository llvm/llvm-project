// Cygwin uses emutls. Clang should pass -femulated-tls to cc1 and cc1 should pass EmulatedTLS to LLVM CodeGen.
// FIXME: Add more targets here to use emutls.
// RUN: %clang -### -std=c++11 -target i686-pc-cygwin %s 2>&1 | FileCheck %s

// CHECK: "-cc1" {{.*}}"-femulated-tls"
