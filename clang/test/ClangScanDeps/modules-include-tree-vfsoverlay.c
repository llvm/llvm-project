// Check include-tree-based caching works with vfsoverlay files.

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json
// RUN: sed -e "s|DIR|%/t|g" %t/vfs.yaml.template > %t/vfs.yaml

// RUN: clang-scan-deps -compilation-database %t/cdb.json -j 1 \
// RUN:   -format experimental-include-tree-full -mode preprocess-dependency-directives \
// RUN:   -cas-path %t/cas > %t/deps.json

// RUN: %deps-to-rsp %t/deps.json --module-name=A > %t/A.rsp
// RUN: %deps-to-rsp %t/deps.json --tu-index 0 > %t/tu.rsp
// RUN: %clang @%t/A.rsp
// RUN: %clang @%t/tu.rsp

// Extract include-tree casids
// RUN: cat %t/A.rsp | sed -E 's|.*"-fcas-include-tree" "(llvmcas://[[:xdigit:]]+)".*|\1|' > %t/A.casid
// RUN: cat %t/tu.rsp | sed -E 's|.*"-fcas-include-tree" "(llvmcas://[[:xdigit:]]+)".*|\1|' > %t/tu.casid

// RUN: echo "MODULE A" > %t/result.txt
// RUN: clang-cas-test -cas %t/cas -print-include-tree @%t/A.casid >> %t/result.txt
// RUN: echo "TRANSLATION UNIT" >> %t/result.txt
// RUN: clang-cas-test -cas %t/cas -print-include-tree @%t/tu.casid >> %t/result.txt

// RUN: FileCheck %s -input-file %t/result.txt -DPREFIX=%/t

// CHECK-LABEL: MODULE A
// CHECK: <module-includes> llvmcas://
// CHECK: 2:1 [[PREFIX]]/elsewhere2/A.h llvmcas://
// CHECK:   Submodule: A
// CHECK: Module Map:
// CHECK: A (framework)
// CHECK: Files:
// CHECK-NOT: modulemap
// CHECK: [[PREFIX]]/elsewhere2/A.h llvmcas://
// CHECK-NOT: modulemap

// CHECK-LABEL: TRANSLATION UNIT
// CHECK: Files:
// CHECK-NOT: .modulemap
// CHECK-NOT: .yaml
// CHECK: [[PREFIX]]/elsewhere2/A.h llvmcas://
// CHECK-NOT: .modulemap
// CHECK-NOT: .yaml

//--- cdb.json.template
[{
  "directory": "DIR",
  "command": "clang -fsyntax-only DIR/tu.c -fmodules -fmodules-cache-path=DIR/module-cache -fimplicit-modules -fimplicit-module-maps -ivfsoverlay DIR/vfs.yaml -F DIR",
  "file": "DIR/tu.c"
}]

//--- vfs.yaml.template
{
  "version": 0,
  "case-sensitive": "false",
  "roots": [
    {
      "name": "DIR/A.framework",
      "type": "directory"
      "contents": [
        {
          "name": "Modules",
          "type": "directory"
          "contents": [
            {
              "external-contents": "DIR/elsewhere1/A.modulemap",
              "name": "module.modulemap",
              "type": "file"
            }
          ]
        },
        {
          "name": "Headers",
          "type": "directory"
          "contents": [
            {
              "external-contents": "DIR/elsewhere2/A.h",
              "name": "A.h",
              "type": "file"
            }
          ]
        }
      ]
    }
  ]
}

//--- elsewhere1/A.modulemap
framework module A { header "A.h" }

//--- elsewhere2/A.h
typedef int A_t;

//--- tu.c
#include "A/A.h"
A_t a = 0;
