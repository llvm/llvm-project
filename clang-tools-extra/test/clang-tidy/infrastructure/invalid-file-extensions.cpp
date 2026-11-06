// RUN: clang-tidy -checks='-*,misc-explicit-constructor' --config='{HeaderFileExtensions: [".h"], ImplementationFileExtensions: [".cpp"]}' %s -- 2>&1 | FileCheck %s
// RUN: clang-tidy -list-checks -checks='-*,misc-explicit-constructor' --config='{HeaderFileExtensions: [".h"]}' 2>&1 | FileCheck --check-prefix=CHECK-LIST %s

// CHECK-DAG: warning: Invalid header file extensions [clang-tidy-config]
// CHECK-DAG: warning: Invalid implementation file extensions [clang-tidy-config]

// CHECK-LIST: Enabled checks:
// CHECK-LIST-NEXT: misc-explicit-constructor

class A { A(int); };
// CHECK-DAG: :[[@LINE-1]]:11: warning: single-argument constructors must be marked explicit
