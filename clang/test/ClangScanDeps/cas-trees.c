// REQUIRES: ondisk_cas
// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json

// RUN: clang-scan-deps -compilation-database %t/cdb.json -cas-path %t/cas -format experimental-tree -mode preprocess-dependency-directives > %t/result1.txt
// RUN: clang-scan-deps -compilation-database %t/cdb.json -cas-path %t/cas -format experimental-tree -mode preprocess > %t/result2.txt
// RUN: diff -u %t/result1.txt %t/result2.txt
// RUN: cat %t/result1.txt | %PathSanitizingFileCheck --sanitize PREFIX=%/t %s

// CHECK:      tree {{.*}} for 'PREFIX{{/|\\}}t1.c'
// CHECK-NEXT: tree {{.*}} for 'PREFIX{{/|\\}}t2.c'

// RUN: clang-scan-deps -compilation-database %t/cdb.json -cas-path %t/cas -format experimental-tree-full -mode preprocess > %t/full_result.json
// RUN: cat %t/full_result.json | %PathSanitizingFileCheck --sanitize PREFIX=%/t --enable-yaml-compatibility %s --check-prefix=FULL-TREE

// FULL-TREE:      {
// FULL-TREE-NEXT:   "modules": [],
// FULL-TREE-NEXT:   "translation-units": [
// FULL-TREE-NEXT:     {
// FULL-TREE:            "cache-key": "[[T1_CACHE_KEY:llvmcas://[[:xdigit:]]+]]"
// FULL-TREE-NEXT:       "casfs-root-id": "[[T1_ROOT_ID:llvmcas://[[:xdigit:]]+]]"
// FULL-TREE-NEXT:       "clang-context-hash": "{{[A-Z0-9]+}}",
// FULL-TREE-NEXT:       "clang-module-deps": [],
// FULL-TREE-NEXT:       "command-line": [
// FULL-TREE:              "-fcas-path"
// FULL-TREE-NEXT:         "PREFIX{{/|\\\\}}cas"
// FULL-TREE:              "-fcas-fs"
// FULL-TREE-NEXT:         "[[T1_ROOT_ID]]"
// FULL-TREE:              "-fcache-compile-job"
// FULL-TREE:            ],
// FULL-TREE:            "file-deps": [
// FULL-TREE-NEXT:         "PREFIX{{/|\\\\}}t1.c",
// FULL-TREE-NEXT:         "PREFIX{{/|\\\\}}top.h",
// FULL-TREE-NEXT:         "PREFIX{{/|\\\\}}n1.h"
// FULL-TREE-NEXT:       ],
// FULL-TREE-NEXT:       "input-file": "PREFIX{{/|\\\\}}t1.c"
// FULL-TREE-NEXT:     }
// FULL-TREE:          {
// FULL-TREE:            "cache-key": "[[T2_CACHE_KEY:llvmcas://[[:xdigit:]]+]]"
// FULL-TREE-NEXT:       "casfs-root-id": "[[T2_ROOT_ID:llvmcas://[[:xdigit:]]+]]"
// FULL-TREE-NEXT:       "clang-context-hash": "{{[A-Z0-9]+}}",
// FULL-TREE-NEXT:       "clang-module-deps": [],
// FULL-TREE-NEXT:       "command-line": [
// FULL-TREE:              "-fcas-path"
// FULL-TREE-NEXT:         "PREFIX{{/|\\\\}}cas"
// FULL-TREE:              "-fcas-fs"
// FULL-TREE-NEXT:         "[[T2_ROOT_ID]]"
// FULL-TREE:              "-fcache-compile-job"
// FULL-TREE:            ],
// FULL-TREE:            "file-deps": [
// FULL-TREE-NEXT:         "PREFIX{{/|\\\\}}t2.c",
// FULL-TREE-NEXT:         "PREFIX{{/|\\\\}}n1.h"
// FULL-TREE-NEXT:       ],
// FULL-TREE-NEXT:       "input-file": "PREFIX{{/|\\\\}}t2.c"
// FULL-TREE-NEXT:     }

// Build with caching
// RUN: %deps-to-rsp %t/full_result.json --tu-index 0 > %t/t1.cc1.rsp
// RUN: %deps-to-rsp %t/full_result.json --tu-index 1 > %t/t2.cc1.rsp
// RUN: %clang @%t/t1.cc1.rsp -Rcompile-job-cache 2> %t/t1-miss.err
// RUN: FileCheck %s -input-file=%t/t1-miss.err -check-prefix=CACHE-MISS
// RUN: %clang @%t/t1.cc1.rsp -Rcompile-job-cache 2>&1 | FileCheck %s -check-prefix=CACHE-HIT
// RUN: %clang @%t/t2.cc1.rsp -Rcompile-job-cache 2> %t/t2-miss.err
// RUN: FileCheck %s -input-file=%t/t2-miss.err -check-prefix=CACHE-MISS
// RUN: %clang @%t/t2.cc1.rsp -Rcompile-job-cache 2>&1 | FileCheck %s -check-prefix=CACHE-HIT
// CACHE-HIT: remark: compile job cache hit
// CACHE-MISS: remark: compile job cache miss

// Check cache keys.
// RUN: cp %t/full_result.json %t/combined.txt
// RUN: cat %t/t1-miss.err >> %t/combined.txt
// RUN: cat %t/t2-miss.err >> %t/combined.txt
// RUN: FileCheck %s -input-file=%t/combined.txt -check-prefix=COMBINED

// COMBINED:        "commands": [
// COMBINED-NEXT:     {
// COMBINED-NEXT:       "cache-key": "[[T1_CACHE_KEY:llvmcas://[[:xdigit:]]+]]"
// COMBINED:          }
// COMBINED:        "commands": [
// COMBINED:          {
// COMBINED-NEXT:       "cache-key": "[[T2_CACHE_KEY:llvmcas://[[:xdigit:]]+]]"
// COMBINED:      remark: compile job cache miss for '[[T1_CACHE_KEY]]'
// COMBINED-NEXT: remark: compile job cache miss for '[[T2_CACHE_KEY]]'

// RUN: clang-scan-deps -compilation-database %t/cdb.json -cas-path %t/cas -format experimental-tree -emit-cas-compdb | %PathSanitizingFileCheck --sanitize PREFIX=%/t --sanitize CLANG=%/clang --enable-yaml-compatibility %s -check-prefix=COMPDB
// COMPDB: [
// COMPDB:   {
// COMPDB:     "file": "PREFIX{{/|\\\\}}t1.c",
// COMPDB:     "directory": "PREFIX",
// COMPDB:     "arguments": [
// COMPDB:       "CLANG",
// COMPDB:       "-cc1",
// COMPDB:       "-fcas-path",
// COMPDB:       "PREFIX{{/|\\\\}}cas",
// COMPDB:       "-fcas-fs",
// COMPDB:   {
// COMPDB:     "file": "PREFIX{{/|\\\\}}t2.c",
// COMPDB:     "directory": "PREFIX",
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
