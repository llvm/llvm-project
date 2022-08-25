// REQUIRES: ondisk_cas
// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json

// RUN: clang-scan-deps -compilation-database %t/cdb.json -cas-path %t/cas -format experimental-tree -mode preprocess-dependency-directives > %t/result1.txt
// RUN: clang-scan-deps -compilation-database %t/cdb.json -cas-path %t/cas -format experimental-tree -mode preprocess > %t/result2.txt
// RUN: diff -u %t/result1.txt %t/result2.txt
// RUN: FileCheck %s -input-file %t/result1.txt -DPREFIX=%/t

// CHECK:      tree {{.*}} for '[[PREFIX]]/t1.c'
// CHECK-NEXT: tree {{.*}} for '[[PREFIX]]/t2.c'

// RUN: clang-scan-deps -compilation-database %t/cdb.json -cas-path %t/cas -format experimental-tree-full -mode preprocess | FileCheck %s -DPREFIX=%/t --check-prefix=FULL-TREE
// FULL-TREE:      {
// FULL-TREE-NEXT:   "modules": [],
// FULL-TREE-NEXT:   "translation-units": [
// FULL-TREE:          {
// FULL-TREE:            "casfs-root-id": "llvmcas://{{[[:xdigit:]]+}}"
// FULL-TREE-NEXT:       "clang-context-hash": "{{[A-Z0-9]+}}",
// FULL-TREE-NEXT:       "clang-module-deps": [],
// FULL-TREE-NEXT:       "command-line": [
// FULL-TREE:            ],
// FULL-TREE:            "file-deps": [
// FULL-TREE-NEXT:         "[[PREFIX]]/t1.c",
// FULL-TREE-NEXT:         "[[PREFIX]]/top.h",
// FULL-TREE-NEXT:         "[[PREFIX]]/n1.h"
// FULL-TREE-NEXT:       ],
// FULL-TREE-NEXT:       "input-file": "[[PREFIX]]/t1.c"
// FULL-TREE:          }
// FULL-TREE:          {
// FULL-TREE:            "casfs-root-id": "llvmcas://{{[[:xdigit:]]+}}"
// FULL-TREE-NEXT:       "clang-context-hash": "{{[A-Z0-9]+}}",
// FULL-TREE-NEXT:       "clang-module-deps": [],
// FULL-TREE-NEXT:       "command-line": [
// FULL-TREE:            ],
// FULL-TREE:            "file-deps": [
// FULL-TREE-NEXT:         "[[PREFIX]]/t2.c",
// FULL-TREE-NEXT:         "[[PREFIX]]/n1.h"
// FULL-TREE-NEXT:       ],
// FULL-TREE-NEXT:       "input-file": "[[PREFIX]]/t2.c"
// FULL-TREE-NEXT:     }

// RUN: clang-scan-deps -compilation-database %t/cdb.json -cas-path %t/cas -format experimental-tree -emit-cas-compdb | FileCheck %s -DPREFIX=%/t -check-prefix=COMPDB
// Check without sharing FileManager.
// RUN: clang-scan-deps -compilation-database %t/cdb.json -cas-path %t/cas -format experimental-tree -reuse-filemanager=0 -emit-cas-compdb | FileCheck %s -DPREFIX=%/t -check-prefix=COMPDB
// COMPDB: [
// COMPDB:   {
// COMPDB:     "file": "[[PREFIX]]/t1.c",
// COMPDB:     "directory": "[[PREFIX]]",
// COMPDB:     "arguments": [
// COMPDB:       "clang",
// COMPDB:       "-cc1",
// COMPDB:       "-fcas-path",
// COMPDB:       "[[PREFIX]]/cas",
// COMPDB:       "-fcas-fs",
// COMPDB:   {
// COMPDB:     "file": "[[PREFIX]]/t2.c",
// COMPDB:     "directory": "[[PREFIX]]",
// COMPDB:     "arguments": [


//--- cdb.json.template
[
  {
    "directory": "DIR",
    "command": "clang -fsyntax-only DIR/t1.c",
    "file": "DIR/t1.c"
  },
  {
    "directory": "DIR",
    "command": "clang -fsyntax-only DIR/t2.c",
    "file": "DIR/t2.c"
  }
]

//--- t1.c
#include "top.h"
#include "n1.h"

//--- t2.c
#include "n1.h"

//--- top.h
#ifndef _TOP_H_
#define _TOP_H_

#define WHATEVER 1
#include "n1.h"

struct S {
  int x;
};

#endif

//--- n1.h
#ifndef _N1_H_
#define _N1_H_

int x1;

#endif
