// Check that a missing VFS errors before trying to scan anything.

// RUN: rm -rf %t && split-file %s %t
// RUN: sed -e "s|DIR|%/t|g" %t/build/cdb.json.in > %t/build/cdb.json
// RUN: not clang-scan-deps -compilation-database %t/build/cdb.json \
// RUN:   -format experimental-full 2>&1 | FileCheck %s

// CHECK: virtual filesystem overlay file
// CHECK: not found

//--- build/cdb.json.in
[{
  "directory": "DIR",
  "command": "clang -c DIR/tu.m -ivfsoverlay DIR/vfs.yaml",
  "file": "DIR/tu.m"
}]

//--- tu.m
