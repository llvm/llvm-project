// Test that failure to create a target when initializing
// CompilerInstanceWithContext (i.e. the user provides a bad target triple)
// is properly diagnosed, instead of continuing to run with Target = nullptr.

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json

// RUN: not clang-scan-deps -compilation-database %t/cdb.json -format \
// RUN:   experimental-full -module-names=M 2>&1 | FileCheck %s

// Check that CompilerInstanceWithContext::initializeOrError properly errors
// during target creation, instead of an assert or a segfault later down the
// line:
// CHECK: Error while scanning dependencies for M:
// CHECK-NEXT: error: unknown target triple 'unknown-unknown-unknown'

//--- module.modulemap
module M { header "M.h" }

//--- M.h
void m(void);

//--- cdb.json.template
[{
  "file": "",
  "directory": "DIR",
  "command": "clang -fmodules -fmodules-cache-path=DIR/cache -I DIR -x c --target=unknown-unknown-unknown"
}]
