// Check cas-fs-based caching works with vfsoverlay files.

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json
// RUN: sed -e "s|DIR|%/t|g" %t/vfs.yaml.template > %t/vfs.yaml

// RUN: clang-scan-deps -compilation-database %t/cdb.json -j 1 \
// RUN:   -format experimental-full -mode preprocess-dependency-directives \
// RUN:   -cas-path %t/cas > %t/deps.json

// RUN: %deps-to-rsp %t/deps.json --module-name=A > %t/A.rsp
// RUN: %deps-to-rsp %t/deps.json --tu-index 0 > %t/tu.rsp
// RUN: %clang @%t/A.rsp
// RUN: %clang @%t/tu.rsp

//--- cdb.json.template
[{
  "directory": "DIR",
  "command": "clang -fsyntax-only DIR/tu.c -fmodules -fmodules-cache-path=DIR/module-cache -fimplicit-modules -fimplicit-module-maps -ivfsoverlay DIR/vfs.yaml",
  "file": "DIR/tu.c"
}]

//--- vfs.yaml.template
{
  "version": 0,
  "case-sensitive": "false",
  "roots": [
    {
      "name": "DIR/A",
      "type": "directory"
      "contents": [
        {
          "external-contents": "DIR/elsewhere1/A.modulemap",
          "name": "module.modulemap",
          "type": "file"
        }
      ]
    }
  ]
}

//--- elsewhere1/A.modulemap
module A { header "A.h" }

//--- A/A.h
typedef int A_t;

//--- tu.c
#include "A/A.h"
A_t a = 0;
