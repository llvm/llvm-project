// Check compiling a header module to a .pcm file.

// RUN: %clang -fmodules-ts -fmodule-name=foobar -x c++-header --precompile %S/Inputs/header1.h %S/Inputs/header2.h %S/Inputs/header3.h -o %t.pcm -v 2>&1 | FileCheck %s --check-prefix=CHECK-PRECOMPILE

// CHECK-PRECOMPILE: -cc1 {{.*}} -emit-header-module
// CHECK-PRECOMPILE-SAME: -fmodules-ts
// CHECK-PRECOMPILE-SAME: -fno-implicit-modules
// CHECK-PRECOMPILE-SAME: -fmodule-name=foobar
// CHECK-PRECOMPILE-SAME: -o {{.*}}.pcm
// CHECK-PRECOMPILE-SAME: -x c++-header
// CHECK-PRECOMPILE-SAME: header1.h
// CHECK-PRECOMPILE-SAME: header2.h
// CHECK-PRECOMPILE-SAME: header3.h

// RUN: %clang -fmodules-ts -fmodule-name=foobar -x c++-header -fsyntax-only %S/Inputs/header1.h %S/Inputs/header2.h %S/Inputs/header3.h -v 2>&1 | FileCheck %s --check-prefix=CHECK-SYNTAX-ONLY
// CHECK-SYNTAX-ONLY: -cc1 {{.*}} -fsyntax-only
// CHECK-SYNTAX-ONLY-SAME: -fmodules-ts
// CHECK-SYNTAX-ONLY-SAME: -fno-implicit-modules
// CHECK-SYNTAX-ONLY-SAME: -fmodule-name=foobar
// CHECK-SYNTAX-ONLY-NOT: -o{{ }}
// CHECK-SYNTAX-ONLY-SAME: -x c++-header
// CHECK-SYNTAX-ONLY-SAME: header1.h
// CHECK-SYNTAX-ONLY-SAME: header2.h
// CHECK-SYNTAX-ONLY-SAME: header3.h
