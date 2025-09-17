// Test that profile summary is set correctly.

// RUN: llvm-profdata merge %S/Inputs/max-function-count.proftext -o %t.profdata
// RUN: %clang_cc1 %s -o - -disable-llvm-passes -emit-llvm -fprofile-instrument-use=clang -fprofile-instrument-use-path=%t.profdata | FileCheck %s
//
int begin(int i) {
  if (i)
    return 0;
  return 1;
}

int end(int i) {
  if (i)
    return 0;
  return 1;
}

int main(int argc, const char *argv[]) {
  begin(0);
  end(1);
  end(1);
  return 0;
}
// CHECK: {{![0-9]+}} = !{i32 1, !"ProfileSummary", {{![0-9]+}}}
// CHECK: {{![0-9]+}} = !{!"DetailedSummary", {{![0-9]+}}}
