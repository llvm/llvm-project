// REQUIRES: case_insensitive_src_dir,ondisk_cas

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s|DIR|%/t|g" %t/cdb1.json.template > %t/cdb1.json
// RUN: sed -e "s|DIR|%/t|g" %t/cdb2.json.template > %t/cdb2.json

// RUN: clang-scan-deps -compilation-database %t/cdb1.json -cas-path %t/cas -format experimental-tree -mode preprocess-dependency-directives > %t/result1.txt
// RUN: clang-scan-deps -compilation-database %t/cdb2.json -cas-path %t/cas -format experimental-tree -mode preprocess > %t/result2.txt
// RUN: sed -e 's/^.*llvmcas/llvmcas/' -e 's/ for.*$//' %t/result1.txt > %t/casid1
// RUN: sed -e 's/^.*llvmcas/llvmcas/' -e 's/ for.*$//' %t/result2.txt > %t/casid2

// RUN: llvm-cas --cas %t/cas --ls-tree-recursive @%t/casid1 | FileCheck -check-prefix=TREE %s -DPREFIX=%/t
// RUN: llvm-cas --cas %t/cas --ls-tree-recursive @%t/casid2 | FileCheck -check-prefix=TREE %s -DPREFIX=%/t

// asdf: FileCheck -check-prefix=TREE %s -input-file %t/result1.txt -DPREFIX=%/t

// TREE: file llvmcas://{{.*}} [[PREFIX]]/Header.h
// TREE: syml llvmcas://{{.*}} [[PREFIX]]/header.h -> Header
// TREE: file llvmcas://{{.*}} [[PREFIX]]/t{{[12]}}.c

//--- cdb1.json.template
[
  {
    "directory": "DIR",
    "command": "clang -fsyntax-only DIR/t1.c",
    "file": "DIR/t1.c"
  }
]

//--- cdb2.json.template
[
  {
    "directory": "DIR",
    "command": "clang -fsyntax-only DIR/t2.c",
    "file": "DIR/t2.c"
  }
]

//--- t1.c
#include "header.h"
#include "Header.h"

void bar1(void) {
  foo();
}

//--- t2.c
#include "Header.h"
#include "header.h"

void bar2(void) {
  foo();
}

//--- header.h
#pragma once
void foo(void);

//--- Header.h
#pragma once
void foo(void);
