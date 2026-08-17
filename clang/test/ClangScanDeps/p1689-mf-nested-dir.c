// UNSUPPORTED: target={{.*}}-aix{{.*}}
//
// When using -format=p1689, clang-scan-deps writes make-style dependency output
// to the path from -MF. The directory where the output file is written may not
// exist yet. Ensure the tool creates missing directories instead of failing or
// aborting.
//
// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
// RUN: sed "s|DIR|%/t|g" %t/cdb.json.in > %t/compile_commands.json
// RUN: clang-scan-deps -compilation-database %t/compile_commands.json -format=p1689 \
// RUN:   -o %t/scan.json
// RUN: cat %t/obj/nested/hello.o.d | sed 's:\\\\\?:/:g' | FileCheck %s --check-prefix=CHECK-DEP
// RUN: cat %t/scan.json | FileCheck %s -DPREFIX=%/t --check-prefix=CHECK-JSON

//--- hello.c
int main(void) { return 0; }

//--- cdb.json.in
[
  {
    "directory": "DIR",
    "command": "clang -c DIR/hello.c -o DIR/obj/nested/hello.o -MMD -MF DIR/obj/nested/hello.o.d",
    "file": "DIR/hello.c",
    "output": "DIR/obj/nested/hello.o"
  }
]

// CHECK-DEP-DAG: hello.o
// CHECK-DEP-DAG: hello.c

// CHECK-JSON-DAG: "primary-output": "[[PREFIX]]/obj/nested/hello.o"
// CHECK-JSON-DAG: "version": 1
