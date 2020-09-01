// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fmodule-name=ImportOnce -fimplicit-module-maps -fmodules-cache-path=%t -F %S/Inputs/import-once %s

// Test #import-ed headers are processed only once, even without header guards.
// Dependency graph is
//
//     Unrelated       ImportOnce
//           ^          ^    ^
//            \        /     |
//       IndirectImporter    |
//                     ^     |
//                      \    |
//                   import-once.m
#import <IndirectImporter/IndirectImporter.h>
#import <ImportOnce/ImportOnce.h>
