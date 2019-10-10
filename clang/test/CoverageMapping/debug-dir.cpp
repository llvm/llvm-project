// %s expands to an absolute path, so to test relative paths we need to create a
// clean directory, put the source there, and cd into it.
// RUN: rm -rf %t
// RUN: mkdir -p %t/foobar
// RUN: cd %t
// RUN: cp %s foobar/debug-dir.cpp

// RUN: %clang_cc1 -fprofile-instrument=clang -fcoverage-mapping -emit-llvm -main-file-name debug-dir.cpp foobar/debug-dir.cpp  -o - | FileCheck -check-prefix=ABSOLUTE %s
//
// ABSOLUTE: @__llvm_coverage_mapping = {{.*"\\01.*clang.*test.*CoverageMapping.*.*foobar.*debug-dir\.cpp}}

// RUN: %clang_cc1 -fprofile-instrument=clang -fcoverage-mapping -emit-llvm -main-file-name debug-dir.cpp foobar/debug-dir.cpp -fdebug-compilation-dir . -o - | FileCheck -check-prefix=RELATIVE %s
//
// RELATIVE: @__llvm_coverage_mapping = {{.*"\\01[^/]*foobar.*debug-dir\.cpp}}

void f1() {}
