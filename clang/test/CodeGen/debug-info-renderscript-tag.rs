// RUN: %clang -emit-llvm -S -g %s -o - | FileCheck %s

// CHECK: !DICompileUnit(language: DW_LANG_GOOGLE_RenderScript{{.*}})
