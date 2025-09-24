// The driver never checks to implicitly enable the explicit module build 
// support unless at least two input files are provided.
// To trigger the C++20 module usage check, we always pass a second dummy file
// as input.
// TODO: Remove -fmodules everywhere once implicitly enabled explicit module 
// builds are supported.

// RUN: split-file %s %t
//--- empty.cpp
// Nothing here

//--- only-global.cpp
// RUN: %clang -std=c++20 -ccc-print-phases -fmodules-driver -Rmodules-driver \
// RUN:   %t/only-global.cpp %t/empty.cpp 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK1
// CHECK1: remark: found C++20 module usage in file '{{.*}}' [-Rmodules-driver]
module;

//--- only-import.cpp
// RUN: %clang -std=c++20 -ccc-print-phases -fmodules-driver -Rmodules-driver \
// RUN:   %t/only-import.cpp %t/empty.cpp 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK2
// CHECK2: remark: found C++20 module usage in file '{{.*}}' [-Rmodules-driver]
import A;

//--- only-export.cpp
// RUN: %clang -std=c++20 -ccc-print-phases -fmodules-driver -Rmodules-driver \
// RUN:   %t/only-export.cpp %t/empty.cpp 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK3
// CHECK3: remark: found C++20 module usage in file '{{.*}}' [-Rmodules-driver]
export module A;

//--- leading-line-comment.cpp
// RUN: %clang -std=c++20 -ccc-print-phases -fmodules-driver -Rmodules-driver \
// RUN:   %t/leading-line-comment.cpp %t/empty.cpp 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK4
// CHECK4: remark: found C++20 module usage in file '{{.*}}' [-Rmodules-driver]
// My line comment
import A;

//--- leading-block-comment1.cpp
// RUN: %clang -std=c++20 -ccc-print-phases -fmodules-driver -Rmodules-driver \
// RUN:   %t/leading-block-comment1.cpp %t/empty.cpp 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK5
// CHECK5: remark: found C++20 module usage in file '{{.*}}' [-Rmodules-driver]
/*My block comment */
import A;

//--- leading-block-comment2.cpp
// RUN: %clang -std=c++20 -ccc-print-phases -fmodules-driver -Rmodules-driver \
// RUN:   %t/leading-block-comment2.cpp %t/empty.cpp 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK6
// CHECK6: remark: found C++20 module usage in file '{{.*}}' [-Rmodules-driver]
/*My line comment */ import A;

//--- inline-block-comment1.cpp
// RUN: %clang -std=c++20 -ccc-print-phases -fmodules-driver -Rmodules-driver \
// RUN:   %t/leading-block-comment1.cpp %t/empty.cpp 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK7
// CHECK7: remark: found C++20 module usage in file '{{.*}}' [-Rmodules-driver]
export/*a comment*/module/*another comment*/A;

//--- inline-block-comment2.cpp
// RUN: %clang -std=c++20 -ccc-print-phases -fmodules-driver -Rmodules-driver \
// RUN:   %t/leading-block-comment2.cpp %t/empty.cpp 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK8
// CHECK8: remark: found C++20 module usage in file '{{.*}}' [-Rmodules-driver]
module/*a comment*/;

//--- leading-directives.cpp
// RUN: %clang -std=c++23 -ccc-print-phases -fmodules-driver -Rmodules-driver \
// RUN:   %t/leading-directives.cpp %t/empty.cpp 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK9
// CHECK9: remark: found C++20 module usage in file '{{.*}}' [-Rmodules-driver]
#define A
#undef A
#if A
#ifdef A
#elifdef A
#elifndef A
#endif
#ifndef A
#elif A
#else
#endif
#endif
#pragma once;
#include <iostream>
import m;

//--- multiline-directive.cpp
// RUN: %clang -std=c++23 -ccc-print-phases -fmodules-driver -Rmodules-driver \
// RUN:   %t/multiline-directive.cpp %t/empty.cpp 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK10
// CHECK10: remark: found C++20 module usage in file '{{.*}}' [-Rmodules-driver]
#define MACRO(a,  \
              b)  \
        call((a), \
             (b)
import a;

//--- leading-line-splice.cpp
// RUN: %clang -std=c++23 -ccc-print-phases -fmodules-driver -Rmodules-driver \
// RUN:   %t/leading-line-splice.cpp %t/empty.cpp 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK11
// CHECK11: remark: found C++20 module usage in file '{{.*}}' [-Rmodules-driver]
\
module;

//--- leading-line-splice-trailing-whitespace.cpp
// RUN: %clang -std=c++23 -ccc-print-phases -fmodules-driver -Rmodules-driver \
// RUN:   %t/leading-line-splice-trailing-whitespace.cpp %t/empty.cpp 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK12
// CHECK12: remark: found C++20 module usage in file '{{.*}}' [-Rmodules-driver]
// v This backslash has trailing whitespace.
   \      
export module A;

//--- comment-line-splice.cpp
// RUN: %clang -std=c++23 -ccc-print-phases -fmodules-driver -Rmodules-driver \
// RUN:   %t/comment-line-splice.cpp %t/empty.cpp 2>&1 \
// RUN:   | FileCheck %s  --allow-empty --check-prefix=CHECK13
// CHECK13-NOT: remark: found C++20 module usage in file '{{.*}}' [-Rmodules-driver]
// My comment continues next-line!\
import A;

//--- comment-line-splice-trailing-whitespace.cpp
// RUN: %clang -std=c++23 -ccc-print-phases -fmodules-driver -Rmodules-driver \
// RUN:   %t/comment-line-splice-trailing-whitespace.cpp %t/empty.cpp 2>&1 \
// RUN:   | FileCheck %s --allow-empty --check-prefix=CHECK14
// CHECK14-NOT: remark: found C++20 module usage in file '{{.*}}' [-Rmodules-driver]
// My comment continues next-line! This backslash has trailing whitespace. -> \   
module;

//--- line-splice-in-directive1.cpp
// RUN: %clang -std=c++23 -ccc-print-phases -fmodules-driver -Rmodules-driver \
// RUN:   %t/line-splice-in-directive1.cpp %t/empty.cpp 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK15
// CHECK15: remark: found C++20 module usage in file '{{.*}}' [-Rmodules-driver]

module\
;

//--- line-splice-in-directive2.cpp
// RUN: %clang -std=c++23 -ccc-print-phases -fmodules-driver -Rmodules-driver \
// RUN:   %t/line-splice-in-directive2.cpp %t/empty.cpp 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK16
// CHECK16: remark: found C++20 module usage in file '{{.*}}' [-Rmodules-driver]

export\
  module\
  A;

//--- no-module-usage1.cpp
// RUN: %clang -std=c++23 -ccc-print-phases -fmodules-driver -Rmodules-driver \
// RUN:   %t/no-module-usage1.cpp %t/empty.cpp 2>&1 \
// RUN:   | FileCheck %s  --allow-empty --check-prefix=CHECK17
// CHECK17-NOT: remark: found C++20 module usage in file '{{.*}}' [-Rmodules-driver]
auto main() -> int {}

//--- no-module-usage2.cpp
// RUN: %clang -std=c++23 -ccc-print-phases -fmodules-driver -Rmodules-driver \
// RUN:   %t/no-module-usage2.cpp %t/empty.cpp 2>&1 \
// RUN:   | FileCheck %s  --allow-empty --check-prefix=CHECK18
// CHECK18-NOT: remark: found C++20 module usage in file '{{.*}}' [-Rmodules-driver]
moduleStruct{};

//--- no-module-usage3.cpp
// RUN: %clang -std=c++23 -ccc-print-phases -fmodules-driver -Rmodules-driver \
// RUN:   %t/no-module-usage3.cpp %t/empty.cpp 2>&1 \
// RUN:   | FileCheck %s  --allow-empty --check-prefix=CHECK19
// CHECK19-NOT: remark: found C++20 module usage in file '{{.*}}' [-Rmodules-driver]
export_struct{};

//--- no-module-usage-namespace-import.cpp
// RUN: %clang -std=c++23 -ccc-print-phases -fmodules-driver -Rmodules-driver \
// RUN:   %t/no-module-usage-namespace-import.cpp %t/empty.cpp 2>&1 \
// RUN:   | FileCheck %s  --allow-empty --check-prefix=CHECK20
// CHECK20-NOT: remark: found C++20 module usage in file '{{.*}}' [-Rmodules-driver]
import::inner xi = {};

//--- no-module-usage-namespace-module.cpp
// RUN: %clang -std=c++23 -ccc-print-phases -fmodules-driver -Rmodules-driver \
// RUN:   %t/no-module-usage-namespace-module.cpp %t/empty.cpp 2>&1 \
// RUN:   | FileCheck %s  --allow-empty --check-prefix=CHECK21
// CHECK21-NOT: remark: found C++20 module usage in file '{{.*}}' [-Rmodules-driver]
module::inner yi = {};

// RUN: not %clang -std=c++20 -ccc-print-phases -fmodules-driver -Rmodules-driver \
// RUN:   imaginary-file.cpp %t/empty.cpp 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NON-EXISTING-FILE-ERR
// CHECK-NON-EXISTING-FILE-ERR: clang: error: no such file or directory: 'imaginary-file.cpp'
