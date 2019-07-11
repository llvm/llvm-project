// RUN: %clang -Xclang -femit-debug-entry-values -g -O2 -Xclang -disable-llvm-passes -S -target x86_64-none-linux-gnu -emit-llvm %s -o - | FileCheck %s -check-prefix=CHECK-ENTRY-VAL-OPT
// CHECK-ENTRY-VAL-OPT: !DILocalVariable(name: "a", arg: 1, scope: {{.*}}, file: {{.*}}, line: {{.*}}, type: {{.*}})
// CHECK-ENTRY-VAL-OPT: !DILocalVariable(name: "b", arg: 2, scope: {{.*}}, file: {{.*}}, line: {{.*}}, type: {{.*}}, flags: DIFlagArgumentNotModified)
//
// For the os_log_helper:
// CHECK-ENTRY-VAL-OPT: !DILocalVariable(name: "buffer", arg: 1, {{.*}}, flags: DIFlagArtificial)
//
// RUN: %clang -g -O2 -Xclang -disable-llvm-passes -target x86_64-none-linux-gnu -S -emit-llvm %s -o - | FileCheck %s
// CHECK-NOT: !DILocalVariable(name: "b", arg: 2, scope: {{.*}}, file: {{.*}}, line: {{.*}}, type: {{.*}}, flags: DIFlagArgumentNotModified)
//
// For the os_log_helper:
// CHECK: !DILocalVariable(name: "buffer", arg: 1, {{.*}}, flags: DIFlagArtificial)

int fn2 (int a, int b) {
  ++a;
  return b;
}

void test_builtin_os_log(void *buf, int i, const char *data) {
  __builtin_os_log_format(buf, "%d %{public}s %{private}.16P", i, data, data);
}
