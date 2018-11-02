// Check compiling a header module to a .pcm file.
//
// RUN: %clang -fmodules-ts -fmodule-name=foobar -x c++-header --precompile %S/Inputs/header1.h %S/Inputs/header2.h %S/Inputs/header3.h -o %t.pcm -v 2>&1 | FileCheck %s --check-prefix=CHECK-PRECOMPILE
//
// CHECK-PRECOMPILE: -cc1 {{.*}} -emit-header-module
// CHECK-PRECOMPILE-SAME: -fmodules-ts
// CHECK-PRECOMPILE-SAME: -fno-implicit-modules
// CHECK-PRECOMPILE-SAME: -fmodule-name=foobar
// CHECK-PRECOMPILE-SAME: -o {{.*}}.pcm
// CHECK-PRECOMPILE-SAME: -x c++
// CHECK-PRECOMPILE-SAME: header1.h
// CHECK-PRECOMPILE-SAME: header2.h
// CHECK-PRECOMPILE-SAME: header3.h
