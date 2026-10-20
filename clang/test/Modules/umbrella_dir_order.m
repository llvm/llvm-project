// RUN: cd %S
// RUN: %clang_cc1 -fmodules -fno-implicit-modules -x objective-c -fmodule-name=x -emit-module Inputs/umbrella_header_order/module.modulemap -o %t/mod.pcm
// RUN: llvm-bcanalyzer --dump --disable-histogram %t/mod.pcm | FileCheck %s

// CHECK: <INPUT_FILE abbrevid=4 op0=1 op1=36 op2=0 op3=0 op4=0 op5=1 op6=1 op7=16/> blob data = 'module.modulemap'
// CHECK: <INPUT_FILE abbrevid=4 op0=2 op1=0 op2=0 op3=0 op4=0 op5=0 op6=0 op7=12/> blob data = 'umbrella{{[/\\]}}A.h'
// CHECK: <INPUT_FILE abbrevid=4 op0=3 op1=0 op2=0 op3=0 op4=0 op5=0 op6=0 op7=12/> blob data = 'umbrella{{[/\\]}}B.h'
// CHECK: <INPUT_FILE abbrevid=4 op0=4 op1=0 op2=0 op3=0 op4=0 op5=0 op6=0 op7=12/> blob data = 'umbrella{{[/\\]}}C.h'
// CHECK: <INPUT_FILE abbrevid=4 op0=5 op1=0 op2=0 op3=0 op4=0 op5=0 op6=0 op7=12/> blob data = 'umbrella{{[/\\]}}D.h'
// CHECK: <INPUT_FILE abbrevid=4 op0=6 op1=0 op2=0 op3=0 op4=0 op5=0 op6=0 op7=12/> blob data = 'umbrella{{[/\\]}}E.h'
// CHECK: <INPUT_FILE abbrevid=4 op0=7 op1=0 op2=0 op3=0 op4=0 op5=0 op6=0 op7=12/> blob data = 'umbrella{{[/\\]}}F.h'
