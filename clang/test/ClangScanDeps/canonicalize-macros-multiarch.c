// Verify cc1 jobs from a multi-arch driver command perform identical macro
// canonicalization.

// REQUIRES: system-darwin
// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s|DIR|%/t|g" %t/cdb.json.in > %t/cdb.json

// RUN: clang-scan-deps -compilation-database %t/cdb.json \
// RUN:   -format experimental-full -optimize-args=canonicalize-macros -j 1 \
// RUN:   > %t/result.json

// RUN: FileCheck %s --input-file %t/result.json
// CHECK-NOT: B=2

//--- cdb.json.in
[{
  "directory": "DIR",
  "command": "clang -c DIR/main.c -o DIR/main.o -arch x86_64 -arch arm64 -DB=2 -DA=1 -UB -DC=3",
  "file": "DIR/main.c"
}]

//--- main.c
int main(void) { return 0; }
