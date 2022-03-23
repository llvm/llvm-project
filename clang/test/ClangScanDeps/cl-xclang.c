// This test checks that '-Xclang' arguments are ignored during the clang-cl command line adjustment.
// This prevents interpreting '-Xclang -I -Xclang /opt/include' as '/o pt/include' (output path).

// RUN: rm -rf %t
// RUN: split-file %s %t

//--- cdb.json.template
[{
  "file": "DIR/test.c",
  "directory": "DIR",
  "command": "clang-cl /c /o DIR/test.o -Xclang -I -Xclang /opt/include -- DIR/test.c"
}]

//--- test.c

// RUN: sed -e "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json
// RUN: clang-scan-deps -compilation-database %t/cdb.json > %t/result.d
// RUN: cat %t/result.d | sed 's:\\\\\?:/:g' | FileCheck %s -DPREFIX=%/t
// CHECK:      [[PREFIX]]/test.o:
// CHECK-NEXT:   [[PREFIX]]/test.c
