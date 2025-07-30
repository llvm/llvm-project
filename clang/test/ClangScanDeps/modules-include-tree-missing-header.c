// Tests that a missing header in a module that is itself imported by another
// module does not crash/assert in the IncludeTreeBuilder.

// RUN: rm -rf %t
// RUN: split-file %s %t

// RUN: not clang-scan-deps -format experimental-include-tree-full -cas-path %t/cas -- %clang -fmodules -fmodules-cache-path=%t/cache -c %t/tu0.m -I%t
// RUN: not clang-scan-deps -format experimental-include-tree-full -cas-path %t/cas -- %clang -fmodules -fmodules-cache-path=%t/cache -c %t/tu1.m -I%t

//--- module.modulemap
module MissingH {
  header "missing.h"
}

module Importer {
  header "importer.h"
}

//--- not-missing.h

//--- importer.h
@import MissingH;

//--- tu0.m
@import MissingH;

//--- tu1.m
@import Importer;
