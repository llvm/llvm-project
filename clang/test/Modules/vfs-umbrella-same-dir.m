// RUN: rm -rf %t
// RUN: split-file %s %t

//--- sources/FW/Private.h
#include <FW/PrivateUnmapped.h>
//--- sources/FW/PrivateUnmapped.h
#include <FW/Public.h>
//--- sources/FW/Public.h
#include <FW/PublicPresent.h>
//--- frameworks/FW.framework/Headers/PublicPresent.h
// empty
//--- frameworks/FW.framework/Modules/module.modulemap
framework module FW { umbrella header "Public.h" }
//--- frameworks/FW.framework/Modules/module.private.modulemap
framework module FW_Private { umbrella header "Private.h" }
//--- vfs.json.in
{
  "case-sensitive": "false",
  "version": 0,
  "roots": [
    {
      "contents": [
        {
          "external-contents": "DIR/sources/FW/Public.h",
          "name": "Public.h",
          "type": "file"
        }
      ],
      "name": "DIR/frameworks/FW.framework/Headers",
      "type": "directory"
    },
    {
      "contents": [
        {
          "external-contents": "DIR/sources/FW/Private.h",
          "name": "Private.h",
          "type": "file"
        }
      ],
      "name": "DIR/frameworks/FW.framework/PrivateHeaders",
      "type": "directory"
    }
  ]
}

//--- tu.m
#import <FW/Private.h>
// expected-no-diagnostics

// RUN: sed -e "s|DIR|%/t|g" %t/vfs.json.in > %t/vfs.json

// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/cache \
// RUN:            -ivfsoverlay %t/vfs.json -I %t/sources -F %t/frameworks -fsyntax-only %t/tu.m -verify
