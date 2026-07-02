//NOTE: this test reuses the existing include files.

// Create PCH with mappend path prefixes
// RUN: mkdir -p %t-dir
// RUN: cp %S/Inputs/pch-through3.h %t-dir
// RUN: cp %S/Inputs/pch-through4.h %t-dir

// RUN: %clang_cc1 -I %S -I %t-dir -x c++-header -emit-pch -fmacro-prefix-map=%S=x:/PREFIX-MAP-IN -fmacro-prefix-map=%t-dir=x:/PREFIX-MAP-OUT \
// RUN:   -o %t.pch %s
// RUN: llvm-bcanalyzer -dump --disable-histogram %t.pch | FileCheck -check-prefix=CHECK-PREFIX-MAP-ORIGINAL %s 
// RUN: llvm-bcanalyzer -dump --disable-histogram %t.pch | FileCheck -check-prefix=CHECK-PREFIX-MAP-IN %s
// RUN: rm -f %t.pch

// RUN: %clang_cc1 -I %S -I %t-dir -x c++-header -emit-pch -fmacro-prefix-map=%S=x:/PREFIX-MAP-IN -fmacro-prefix-map=%t-dir=x:/PREFIX-MAP-OUT \
// RUN:   -include %S/cmdline-include1.h -include cmdline-include2.h -o %t.pch %s
// RUN: llvm-bcanalyzer -dump --disable-histogram %t.pch | FileCheck -check-prefix=CHECK-PREFIX-MAP-ORIGINAL %s 
// RUN: llvm-bcanalyzer -dump --disable-histogram %t.pch | FileCheck -check-prefix=CHECK-PREFIX-MAP-IN-CMD %s

// Use PCH with mapped path prefixes.
// RUN: %clang_cc1 -I %S -I %t-dir -include-pch %t.pch -fmacro-prefix-map=%S=x:/PREFIX-MAP-IN -fmacro-prefix-map=%t-dir=x:/PREFIX-MAP-OUT %s
// RUN: rm -f %t.pch

// RUN: %clang_cl -Werror /I %S /I %t-dir /Yc%s /FI%s /c /Fo%t.pch.obj /Fp%t.pch /pathmap:%S=x:/PREFIX-MAP-IN /pathmap:%t-dir=x:/PREFIX-MAP-OUT -- %s
// RUN: llvm-bcanalyzer -dump --disable-histogram %t.pch | FileCheck -check-prefix=CHECK-PREFIX-MAP-ORIGINAL %s 
// RUN: llvm-bcanalyzer -dump --disable-histogram %t.pch | FileCheck -check-prefix=CHECK-PREFIX-MAP-IN %s

// use {{[/\\]}} to follow LLVM_WINDOWS_PREFER_FORWARD_SLASH=ON|OFF on Windows.

// CHECK-PREFIX-MAP-ORIGINAL: <ORIGINAL_FILE {{.*}}/> blob data = 'x:/PREFIX-MAP-IN{{[/\\]}}pch-path-prefix-map.cpp'
// CHECK-PREFIX-MAP-IN: <INPUT_FILE {{.*}}/> blob data = 'x:/PREFIX-MAP-IN{{[/\\]}}pch-path-prefix-map.cpp'
// CHECK-PREFIX-MAP-IN: <INPUT_FILE {{.*}}/> blob data = 'x:/PREFIX-MAP-IN{{[/\\]}}Inputs{{[/\\]}}pch-through{{[12]}}.h'
// CHECK-PREFIX-MAP-IN: <INPUT_FILE {{.*}}/> blob data = 'x:/PREFIX-MAP-IN{{[/\\]}}Inputs{{[/\\]}}pch-through{{[12]}}.h'
// CHECK-PREFIX-MAP-IN: <INPUT_FILE {{.*}}/> blob data = 'x:/PREFIX-MAP-OUT{{[/\\]}}pch-through{{[34]}}.h'
// CHECK-PREFIX-MAP-IN: <INPUT_FILE {{.*}}/> blob data = 'x:/PREFIX-MAP-OUT{{[/\\]}}pch-through{{[34]}}.h'

// CHECK-PREFIX-MAP-IN-CMD: <INPUT_FILE {{.*}}/> blob data = 'x:/PREFIX-MAP-IN{{[/\\]}}pch-path-prefix-map.cpp'
// CHECK-PREFIX-MAP-IN-CMD: <INPUT_FILE {{.*}}/> blob data = 'x:/PREFIX-MAP-IN{{[/\\]}}cmdline-include{{[12]}}.h'
// CHECK-PREFIX-MAP-IN-CMD: <INPUT_FILE {{.*}}/> blob data = 'x:/PREFIX-MAP-IN{{[/\\]}}cmdline-include{{[12]}}.h'
// CHECK-PREFIX-MAP-IN-CMD: <INPUT_FILE {{.*}}/> blob data = 'x:/PREFIX-MAP-IN{{[/\\]}}Inputs{{[/\\]}}pch-through{{[12]}}.h'
// CHECK-PREFIX-MAP-IN-CMD: <INPUT_FILE {{.*}}/> blob data = 'x:/PREFIX-MAP-IN{{[/\\]}}Inputs{{[/\\]}}pch-through{{[12]}}.h'
// CHECK-PREFIX-MAP-IN-CMD: <INPUT_FILE {{.*}}/> blob data = 'x:/PREFIX-MAP-OUT{{[/\\]}}pch-through{{[34]}}.h'
// CHECK-PREFIX-MAP-IN-CMD: <INPUT_FILE {{.*}}/> blob data = 'x:/PREFIX-MAP-OUT{{[/\\]}}pch-through{{[34]}}.h'

#include "Inputs/pch-through1.h"
#include "Inputs/pch-through2.h"
#include "pch-through3.h"
#include <pch-through4.h>

