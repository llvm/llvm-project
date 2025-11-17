// REQUIRES: ondisk_cas
// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json

// RUN: %clang -x c-header %t/prefix.h -target x86_64-apple-macos12 -o %t/prefix.pch -fdepscan=inline -Xclang -fcas-path -Xclang %t/cas
// RUN: clang-scan-deps -compilation-database %t/cdb.json -format experimental-include-tree -cas-path %t/cas > %t/result.txt
// RUN: cat %t/result.txt | %PathSanitizingFileCheck --sanitize PREFIX=%/t %s

// CHECK:      {{.*}} - PREFIX{{/|\\}}t.c
// CHECK-NEXT: (PCH)
// CHECK-NEXT: PREFIX{{/|\\}}t.c
// CHECK-NEXT: 1:1 <built-in>
// CHECK-NEXT: PREFIX{{/|\\}}t.h
// CHECK-NEXT: Files:
// CHECK-NEXT: PREFIX{{/|\\}}t.c
// CHECK-NEXT: PREFIX{{/|\\}}t.h
// CHECK-NEXT: PREFIX{{/|\\}}prefix.h
// CHECK-NEXT: PREFIX{{/|\\}}n1.h
// CHECK-NEXT: PREFIX{{/|\\}}n2.h
// CHECK-NEXT: PREFIX{{/|\\}}n3.h
// CHECK-NOT: PREFIX

// RUN: clang-scan-deps -compilation-database %t/cdb.json -format experimental-include-tree-full -cas-path %t/cas > %t/deps.json

// RUN: cat %t/result.txt > %t/full.txt
// RUN: echo "FULL DEPS START" >> %t/full.txt
// RUN: cat %t/deps.json >> %t/full.txt

// RUN: cat %t/full.txt | %PathSanitizingFileCheck --sanitize PREFIX=%/t --sanitize CLANG=%/clang --enable-yaml-compatibility -check-prefix=FULL %s

// Capture the tree id from experimental-include-tree ; ensure that it matches
// the result from experimental-full.
// FULL: [[TREE_ID:llvmcas://[[:xdigit:]]+]] - PREFIX{{/|\\}}t.c
// FULL: FULL DEPS START

// FULL-NEXT: {
// FULL-NEXT:   "modules": []
// FULL-NEXT:   "translation-units": [
// FULL-NEXT:     {
// FULL-NEXT:       "commands": [
// FULL-NEXT:         {
// FULL:                "clang-module-deps": []
// FULL:                "command-line": [
// FULL-NEXT:             "-cc1"
// FULL:                  "-fcas-path"
// FULL-NEXT:             "PREFIX{{/|\\\\}}cas"
// FULL:                  "-disable-free"
// FULL:                  "-fcas-include-tree"
// FULL-NEXT:             "[[TREE_ID]]"
// FULL:                  "-fcache-compile-job"
// FULL:                  "-fsyntax-only"
// FULL:                  "-x"
// FULL-NEXT:             "c"
// FULL-NOT:              "t.c"
// FULL:                  "-main-file-name"
// FULL-NEXT:             "t.c"
// FULL-NOT:              "t.c"
// FULL:                ]
// FULL:                "executable": "CLANG"
// FULL:                "file-deps": [
// FULL-NEXT:             "PREFIX{{/|\\\\}}t.c"
// FULL-NEXT:             "PREFIX{{/|\\\\}}t.h"
// FULL-NEXT:             "PREFIX{{/|\\\\}}prefix.pch"
// FULL-NEXT:           ]
// FULL:                "input-file": "PREFIX{{/|\\\\}}t.c"
// FULL:              }
// FULL:            ]
// FULL:          }
// FULL:        ]
// FULL:      }

// Build the include-tree command
// RUN: %deps-to-rsp %t/deps.json --tu-index 0 > %t/tu.rsp
// RUN: %clang @%t/tu.rsp

//--- prefix.h
#include "n1.h"
#import "n2.h"
#include "n3.h"

//--- n1.h
#ifndef _N1_H_
#define _N1_H_

int n1 = 0;

#endif

//--- n2.h
int n2 = 0;

//--- n3.h
#pragma once
int n3 = 0;

//--- cdb.json.template
[{
  "directory" : "DIR",
  "command" : "clang -fsyntax-only DIR/t.c -target x86_64-apple-macos12 -isysroot DIR -Xclang -include-pch -Xclang DIR/prefix.pch",
  "file" : "DIR/t.c"
}]

//--- t.c
#include "t.h"
// Should not include due to macro guard.
#include "n1.h"
// Should not include due to #import from 'prefix.h'.
#include "n2.h"
// Should not include due to '#pragma once'
#include "n3.h"

//--- t.h
