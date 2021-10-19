// This test checks if a symbol gets mangled dwarf names with -fpseudo-probe-for-profiling option.
// RUN: %clang_cc1 -triple x86_64 -x c++ -S -emit-llvm -debug-info-kind=line-tables-only -o - < %s | FileCheck %s --check-prefix=PLAIN
// RUN: %clang_cc1 -triple x86_64 -x c++  -S -emit-llvm -debug-info-kind=line-tables-only -fpseudo-probe-for-profiling -o - < %s | FileCheck %s --check-prefix=MANGLE

int foo() {
  return 0;
}

// PLAIN: define dso_local i32 @_Z3foov()
// PLAIN: distinct !DISubprogram(name: "foo", scope:
// MANGLE: define dso_local i32 @_Z3foov()
// MANGLE: distinct !DISubprogram(name: "foo", linkageName: "_Z3foov"
