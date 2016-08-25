// RUN: %clang_cc1 -debug-info-kind=limited -split-dwarf-file foo.dwo -S -emit-llvm -o - %s | FileCheck %s
int main (void) {
  return 0;
}

// Testing to ensure that the dwo name gets output into the compile unit.
// CHECK: !DICompileUnit({{.*}}, splitDebugFilename: "foo.dwo"
