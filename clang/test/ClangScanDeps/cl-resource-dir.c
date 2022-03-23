// This test checks that the clang-cl compiler is correctly invoked to deduce resource directory.

// REQUIRES: shell

// RUN: rm -rf %t
// RUN: split-file %s %t

//--- cdb.json.template
[{
  "file": "DIR/test.c",
  "directory": "DIR",
  "command": "DIR/clang-cl /c /o DIR/test.o -- DIR/test.c"
}]

//--- clang-cl
#!/bin/sh

# This is a fake compiler that should be invoked the clang-cl way to print the resource directory.

if [ "$1" = "/clang:-print-resource-dir" ]; then
  echo "/pass"
else
  echo "/fail"
fi;

//--- test.c

// RUN: chmod +x %t/clang-cl
// RUN: sed -e "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json
// RUN: clang-scan-deps -compilation-database %t/cdb.json --resource-dir-recipe invoke-compiler \
// RUN:   --format experimental-full > %t/result.json
// RUN: cat %t/result.json | sed 's:\\\\\?:/:g' | FileCheck %s
// CHECK:      "-resource-dir"
// CHECK-NEXT: "/pass"
