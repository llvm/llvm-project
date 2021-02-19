// RUN: %clang_cc1 -mllvm -emptyline-comment-coverage=false -fprofile-instrument=clang -fcoverage-mapping -mllvm -enable-name-compression=false -emit-llvm -main-file-name abspath.cpp %S/Inputs/../abspath.cpp -o - | FileCheck -check-prefix=RMDOTS %s

// RMDOTS: @__llvm_coverage_mapping = {{.*}}"\01
// RMDOTS-NOT: Inputs
// RMDOTS: "

// RUN: mkdir -p %t/test && cd %t/test
// RUN: echo "void f1() {}" > f1.c
// RUN: %clang_cc1 -mllvm -emptyline-comment-coverage=false -fprofile-instrument=clang -fcoverage-mapping -mllvm -enable-name-compression=false -emit-llvm -main-file-name abspath.cpp ../test/f1.c -o - | FileCheck -check-prefix=RELPATH %s

// RELPATH: @__llvm_coverage_mapping = {{.*}}"\01
// RELPATH: {{[/\\].*(/|\\\\)test(/|\\\\)f1}}.c
// RELPATH: "

void f1() {}
