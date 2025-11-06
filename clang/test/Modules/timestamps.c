/// Verify timestamps that gets embedded in the module
#include <c-header.h>

// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fdisable-module-hash \
// RUN:   -fmodules-cache-path=%t -I %S/Inputs %s
// RUN: cp %t/c_library.pcm %t1.pcm
// RUN: llvm-bcanalyzer --dump --disable-histogram %t1.pcm > %t1.dump
// RUN: FileCheck %s --check-prefix=CHECK --check-prefix=TIMESTAMP --input-file %t1.dump
// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fdisable-module-hash \
// RUN:   -fmodules-cache-path=%t -I %S/Inputs -fno-pch-timestamp %s
// RUN: cp %t/c_library.pcm %t2.pcm
// RUN: llvm-bcanalyzer --dump --disable-histogram %t2.pcm > %t2.dump
// RUN: FileCheck %s --check-prefix=CHECK --check-prefix=NOTIMESTAMP --input-file %t2.dump
// RUN: not diff %t1.dump %t2.dump


// CHECK: <INPUT_FILES_BLOCK NumWords=[[#]] BlockCodeSize=[[#]]>
// CHECK-NEXT: <INPUT_FILE abbrevid=4 op0=1 op1=[[#]]
// TIMESTAMP-NOT: op2=0
// NOTIMESTAMP: op2=0
// CHECK-SAME: blob data = 'module.modulemap'
// CHECK-NEXT: <INPUT_FILE_HASH abbrevid=[[#]] op0=[[#]] op1=[[#]]/>
// CHECK-NEXT: <INPUT_FILE abbrevid=4 op0=2 op1=[[#]]
// TIMESTAMP-NOT: op2=0
// NOTIMESTAMP: op2=0
// CHECK-SAME: blob data = 'c-header.h'
// CHECK-NEXT: <INPUT_FILE_HASH abbrevid=[[#]] op0=[[#]] op1=[[#]]/>
// CHECK-NEXT: </INPUT_FILES_BLOCK>
