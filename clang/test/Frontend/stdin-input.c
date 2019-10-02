// RUN: cat %s | %clang -emit-llvm -g -S \
// RUN: -Xclang -main-file-name -Xclang test/foo.c -x c - -o - | FileCheck %s
// CHECK: ; ModuleID = 'test/foo.c'
// CHECK: source_filename = "test/foo.c"
// CHECK: !DIFile(filename: "test/foo.c"

int main() {}
