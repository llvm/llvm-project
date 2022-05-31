// RUN: rm -rf %t
// RUN: mkdir %t

// Check compiling a module interface to a .pcm file.
//
// RUN: %clang -std=c++2a -x c++-module --precompile %s -o %t/module.pcm -v 2>&1 | FileCheck %s --check-prefix=CHECK-PRECOMPILE
//
// CHECK-PRECOMPILE: -cc1 {{.*}} -emit-module-interface
// CHECK-PRECOMPILE-SAME: -o {{.*}}.pcm
// CHECK-PRECOMPILE-SAME: -x c++
// CHECK-PRECOMPILE-SAME: modules.cpp

// Check compiling a .pcm file to a .o file.
//
// RUN: %clang -std=c++2a %t/module.pcm -S -o %t/module.pcm.o -v 2>&1 | FileCheck %s --check-prefix=CHECK-COMPILE
//
// CHECK-COMPILE: -cc1 {{.*}} {{-emit-obj|-S}}
// CHECK-COMPILE-SAME: -o {{.*}}module{{2*}}.pcm.o
// CHECK-COMPILE-SAME: -x pcm
// CHECK-COMPILE-SAME: {{.*}}.pcm

// Check use of a .pcm file in another compilation.
//
// RUN: %clang -std=c++2a -fmodule-file=%t/module.pcm -Dexport= %s -S -o %t/module.o -v 2>&1 | FileCheck %s --check-prefix=CHECK-USE
// RUN: %clang -std=c++20 -fmodule-file=%t/module.pcm -Dexport= %s -S -o %t/module.o -v 2>&1 | FileCheck %s --check-prefix=CHECK-USE
//
// CHECK-USE: -cc1
// CHECK-USE-SAME: {{-emit-obj|-S}}
// CHECK-USE-SAME: -fmodule-file={{.*}}.pcm
// CHECK-USE-SAME: -o {{.*}}.{{o|s}}{{"?}} {{.*}}-x c++
// CHECK-USE-SAME: modules.cpp

// Check combining precompile and compile steps works.
//
// RUN: %clang -std=c++2a -x c++-module %s -S -o %t/module2.pcm.o -v 2>&1 | FileCheck %s --check-prefix=CHECK-PRECOMPILE --check-prefix=CHECK-COMPILE

// Check that .cppm is treated as a module implicitly.
//
// RUN: cp %s %t/module.cppm
// RUN: %clang -std=c++2a --precompile %t/module.cppm -o %t/module.pcm -v 2>&1 | FileCheck %s --check-prefix=CHECK-PRECOMPILE

// Check compiling a header unit to a .pcm file.
//
// RUN: echo '#define FOO BAR' > %t/foo.h
// RUN: %clang -std=c++2a --precompile -x c++-header %t/foo.h -fmodule-name=header -o %t/foo.pcm -v 2>&1 | FileCheck %s --check-prefix=CHECK-HEADER-UNIT
//
// CHECK-HEADER-UNIT: -cc1
// CHECK-HEADER-UNIT-SAME: -emit-header-module
// CHECK-HEADER-UNIT-SAME: -fmodule-name=header
// CHECK-HEADER-UNIT-SAME: -o {{.*}}foo.pcm
// CHECK-HEADER-UNIT-SAME: -x c++-header
// CHECK-HEADER-UNIT-SAME: foo.h

// Check use of header unit.
//
// RUN: %clang -std=c++2a -fmodule-file=%t/module.pcm -fmodule-file=%t/foo.pcm -I%t -DIMPORT -Dexport= %s -E -o - -v 2>&1 | FileCheck %s --check-prefix=CHECK-HEADER-UNIT-USE
//
// CHECK-HEADER-UNIT-USE: -cc1
// CHECK-HEADER-UNIT-USE: -E
// CHECK-HEADER-UNIT-USE: -fmodule-file={{.*}}module.pcm
// CHECK-HEADER-UNIT-USE: -fmodule-file={{.*}}foo.pcm

// Note, we use -Dexport= to make this a module implementation unit when building the implementation.
export module foo;

#ifdef IMPORT
// CHECK-HEADER-UNIT-USE: FOO;
FOO;

// CHECK-HEADER-UNIT-USE: import header.{{.*}}foo.h{{.*}};
import "foo.h";

// CHECK-HEADER-UNIT-USE: BAR;
FOO;
#endif

// Check the independent use of -fcxx-modules
//
// RUN: %clang -fcxx-modules -std=c++17 -### %s 2>&1 | FileCheck %s --check-prefix=CHECK-CXX17-MODULES
// CHECK-CXX17-MODULES: "-fcxx-modules"
// RUN: %clang -fcxx-modules -std=c++14 -### %s 2>&1 | FileCheck %s --check-prefix=CHECK-CXX14-MODULES
// CHECK-CXX14-MODULES: "-fcxx-modules"
// RUN: %clang -fcxx-modules -std=c++11 -### %s 2>&1 | FileCheck %s --check-prefix=CHECK-CXX11-MODULES
// CHECK-CXX11-MODULES: "-fcxx-modules"
// RUN: %clang -fcxx-modules -std=c++03 -### %s 2>&1 | FileCheck %s --check-prefix=CHECK-CXX03-MODULES
// CHECK-CXX03-MODULES: "-fcxx-modules"
