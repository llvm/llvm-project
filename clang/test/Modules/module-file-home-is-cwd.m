// UNSUPPORTED: target={{.*}}-zos{{.*}}, target={{.*}}-aix{{.*}}
// RUN: cd %S
// RUN: %clang_cc1 -x objective-c -fmodules -fno-implicit-modules \
// RUN:     -fmodule-file-home-is-cwd -fmodule-name=libA -emit-module \
// RUN:     -fmodules-embed-all-files %S/Inputs/normal-module-map/module.map \
// RUN:     -o %t/mod.pcm
// RUN: llvm-bcanalyzer --dump --disable-histogram %t/mod.pcm | FileCheck %s
// RUN: llvm-bcanalyzer --dump --disable-histogram %t/mod.pcm | FileCheck %s --check-prefix=INPUT

// CHECK-NOT: MODULE_DIRECTORY
// INPUT: <INPUT_FILE {{.*}}/> blob data = 'Inputs{{/|\\}}normal-module-map{{/|\\}}module.map'
// INPUT: <INPUT_FILE {{.*}}/> blob data = 'Inputs{{/|\\}}normal-module-map{{/|\\}}a2.h'
// INPUT: <INPUT_FILE {{.*}}/> blob data = 'Inputs{{/|\\}}normal-module-map{{/|\\}}a1.h'

@import libA;

// RUN: cd %t
// RUN: %clang_cc1 -x objective-c -fmodules -fno-implicit-modules -debug-info-kind=limited \
// RUN:     -debugger-tuning=lldb -dwarf-ext-refs -fmodule-file-home-is-cwd \
// RUN:     -fmodule-map-file=%S/Inputs/normal-module-map/module.map \
// RUN:     -fmodule-file=libA=mod.pcm -emit-llvm -o %t-mod.ll %s
// RUN: cat %t-mod.ll | FileCheck %s --check-prefix=SKELETON

// SKELETON: !DICompileUnit(language: DW_LANG_ObjC, {{.*}}, splitDebugFilename: "mod.pcm"
