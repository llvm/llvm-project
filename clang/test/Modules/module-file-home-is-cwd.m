// RUN: cd %S
// RUN: %clang_cc1 -fmodules -fno-implicit-modules -fmodule-file-home-is-cwd -fmodule-name=libA -emit-module Inputs/normal-module-map/module.map -o %t/mod.pcm
// RUN: llvm-bcanalyzer --dump --disable-histogram %t/mod.pcm | FileCheck %s

// CHECK: <INPUT_FILE {{.*}}/> blob data = 'Inputs{{/|\\}}normal-module-map{{/|\\}}a1.h'
// CHECK: <INPUT_FILE {{.*}}/> blob data = 'Inputs{{/|\\}}normal-module-map{{/|\\}}a2.h'
// CHECK: <INPUT_FILE {{.*}}/> blob data = 'Inputs{{/|\\}}normal-module-map{{/|\\}}module.map'
// CHECK-NOT: MODULE_DIRECTORY
