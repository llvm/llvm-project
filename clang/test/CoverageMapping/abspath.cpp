// RUN: %clang_cc1 -mllvm -emptyline-comment-coverage=false -fprofile-instrument=clang -fcoverage-mapping -mllvm -enable-name-compression=false -emit-llvm -main-file-name abspath.cpp %S/Inputs/../abspath.cpp -o - | FileCheck -check-prefix=RMDOTS %s

// RMDOTS: @__llvm_coverage_mapping = {{.*}}"\02
// RMDOTS-NOT: Inputs
// RMDOTS: "

// RUN: mkdir -p %t/test && cd %t/test
// RUN: echo "void f1(void) {}" > f1.c
// RUN: %clang_cc1 -mllvm -emptyline-comment-coverage=false -fprofile-instrument=clang -fcoverage-mapping -mllvm -enable-name-compression=false -emit-llvm -main-file-name abspath.cpp %t/test/f1.c -o - | FileCheck -check-prefix=ABSPATH %s

// RELPATH: @__llvm_coverage_mapping = {{.*}}"\02
// RELPATH: {{..(/|\\\\)test(/|\\\\)f1}}.c
// RELPATH: "

// ABSPATH: @__llvm_coverage_mapping = {{.*}}"\02
// ABSPATH: {{[/\\].*(/|\\\\)test(/|\\\\)f1}}.c
// ABSPATH: "

void f1(void) {}
