// This test checks that the output path is correctly deduced/recognized in clang-cl mode.

// RUN: rm -rf %t
// RUN: split-file %s %t

//--- deduce-cdb.json.template
[{
  "file": "DIR/test.c",
  "directory": "DIR",
  "command": "clang --driver-mode=cl /c -- DIR/test.c"
},{
  "file": "DIR/test.c",
  "directory": "DIR",
  "command": "clang-cl /c -- DIR/test.c"
}]

//--- recognize-cdb.json.template
[{
  "file": "DIR/test.c",
  "directory": "DIR",
  "command": "clang-cl /c -o DIR/test.o -- DIR/test.c"
},{
  "file": "DIR/test.c",
  "directory": "DIR",
  "command": "clang-cl /c /o DIR/test.o -- DIR/test.c"
},{
  "file": "DIR/test.c",
  "directory": "DIR",
  "command": "clang-cl /c -oDIR/test.o -- DIR/test.c"
},{
  "file": "DIR/test.c",
  "directory": "DIR",
  "command": "clang-cl /c /oDIR/test.o -- DIR/test.c"
},{
  "file": "DIR/test.c",
  "directory": "DIR",
  "command": "clang-cl /c -FoDIR/test.o -- DIR/test.c"
},{
  "file": "DIR/test.c",
  "directory": "DIR",
  "command": "clang-cl /c /FoDIR/test.o -- DIR/test.c"
}]

//--- last-arg-cdb.json.template
[{
  "file": "DIR/test.c",
  "directory": "DIR",
  "command": "clang-cl /c -o DIR/test.o -o DIR/last.o -- DIR/test.c"
}]

//--- test.c

// Check that missing output path is deduced (with both clang-cl executable and driver mode flag):
//
// RUN: sed -e "s|DIR|%/t|g" %t/deduce-cdb.json.template > %t/deduce-cdb.json
// RUN: clang-scan-deps -compilation-database %t/deduce-cdb.json -j 1 > %t/deduce-result.d
// RUN: cat %t/deduce-result.d | sed 's:\\\\\?:/:g' | FileCheck %s -DPREFIX=%/t --check-prefix=CHECK-DEDUCE
// CHECK-DEDUCE:      test.obj:
// CHECK-DEDUCE-NEXT:   [[PREFIX]]/test.c
// CHECK-DEDUCE-NEXT: test.obj:
// CHECK-DEDUCE-NEXT:   [[PREFIX]]/test.c

// Check that all the different ways to specify output file are recognized:
//
// RUN: sed -e "s|DIR|%/t|g" %t/recognize-cdb.json.template > %t/recognize-cdb.json
// RUN: clang-scan-deps -compilation-database %t/recognize-cdb.json -j 1 > %t/recognize-result.d
// RUN: cat %t/recognize-result.d | sed 's:\\\\\?:/:g' | FileCheck %s -DPREFIX=%/t --check-prefix=CHECK-RECOGNIZE
// CHECK-RECOGNIZE:      [[PREFIX]]/test.o:
// CHECK-RECOGNIZE-NEXT:   [[PREFIX]]/test.c
// CHECK-RECOGNIZE-NEXT: [[PREFIX]]/test.o:
// CHECK-RECOGNIZE-NEXT:   [[PREFIX]]/test.c
// CHECK-RECOGNIZE-NEXT: [[PREFIX]]/test.o:
// CHECK-RECOGNIZE-NEXT:   [[PREFIX]]/test.c
// CHECK-RECOGNIZE-NEXT: [[PREFIX]]/test.o:
// CHECK-RECOGNIZE-NEXT:   [[PREFIX]]/test.c
// CHECK-RECOGNIZE-NEXT: [[PREFIX]]/test.o:
// CHECK-RECOGNIZE-NEXT:   [[PREFIX]]/test.c
// CHECK-RECOGNIZE-NEXT: [[PREFIX]]/test.o:
// CHECK-RECOGNIZE-NEXT:   [[PREFIX]]/test.c

// Check that the last argument specifying the output path wins.
//
// RUN: sed -e "s|DIR|%/t|g" %t/last-arg-cdb.json.template > %t/last-arg-cdb.json
// RUN: clang-scan-deps -compilation-database %t/last-arg-cdb.json > %t/last-arg-result.d
// RUN: cat %t/last-arg-result.d | sed 's:\\\\\?:/:g' | FileCheck %s -DPREFIX=%/t --check-prefix=CHECK-LAST
// CHECK-LAST:      [[PREFIX]]/last.o:
// CHECK-LAST-NEXT:   [[PREFIX]]/test.c
