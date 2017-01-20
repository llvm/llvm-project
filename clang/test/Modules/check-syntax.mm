// RUN: not %clang -fmodules -fno-cxx-modules -fsyntax-only %s 2>&1 | FileCheck %s
// rdar://19399671

// CHECK: use of '@import' when C++ modules are disabled
@import Foundation;
