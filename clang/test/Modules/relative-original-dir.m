// RUN: rm -rf %t/normal-module-map
// RUN: mkdir -p %t
// RUN: cp -r %S/Inputs/normal-module-map %t
// RUN: %clang_cc1 -fmodules -fno-implicit-modules -fmodule-name=libA -emit-module %t/normal-module-map/module.map -o %t/normal-module-map/outdir/mod.pcm
// RUN: llvm-bcanalyzer --dump --disable-histogram %t/normal-module-map/outdir/mod.pcm | FileCheck %s

// CHECK: <ORIGINAL_PCH_DIR abbrevid=7/> blob data = 'outdir'
