// This test checks that the module map paths we're reporting are the as-requested
// paths (as opposed to the paths files resolve to after going through VFS overlays).

// RUN: rm -rf %t
// RUN: split-file %s %t

//--- real/module.modulemap
framework module FW { header "Header.h" }
//--- real/Header.h
//--- overlay.json.template
{
  "case-sensitive": "false",
  "version": "0",
  "roots": [
    {
      "contents": [
        {
          "external-contents" : "DIR/real/Header.h",
          "name" : "Header.h",
          "type" : "file"
        }
      ],
      "name": "DIR/frameworks/FW.framework/Headers",
      "type": "directory"
    },
    {
      "contents": [
        {
          "external-contents": "DIR/real/module.modulemap",
          "name": "module.modulemap",
          "type": "file"
        }
      ],
      "name": "DIR/frameworks/FW.framework/Modules",
      "type": "directory"
    }
  ]
}

//--- modules/module.modulemap
module Importer { header "header.h" }
//--- modules/header.h
#include <FW/Header.h>

//--- cdb.json.template
[{
  "file": "DIR/tu.m",
  "directory": "DIR",
  "command": "clang -fmodules -fmodules-cache-path=DIR/cache -Werror=non-modular-include-in-module -ivfsoverlay DIR/overlay.json -F DIR/frameworks -I DIR/modules -c DIR/tu.m -o DIR/tu.o"
}]

//--- tu.m
@import Importer;

// RUN: sed -e "s|DIR|%/t|g" %t/overlay.json.template > %t/overlay.json
// RUN: sed -e "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json
// RUN: clang-scan-deps -compilation-database %t/cdb.json -format experimental-full > %t/result.json

// RUN: %deps-to-rsp %t/result.json --module-name=FW > %t/FW.cc1.rsp
// RUN: %deps-to-rsp %t/result.json --module-name=Importer > %t/Importer.cc1.rsp
// RUN: %deps-to-rsp %t/result.json --tu-index=0 > %t/tu.rsp
// RUN: %clang @%t/FW.cc1.rsp
// RUN: %clang @%t/Importer.cc1.rsp
// RUN: %clang @%t/tu.rsp
