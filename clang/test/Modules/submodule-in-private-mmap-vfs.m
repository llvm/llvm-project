// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s|DIR|%/t|g" %t/vfs.json.in > %t/vfs.json
// RUN: %clang_cc1 -fmodules -fno-modules-share-filemanager -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t -I%t/Virtual -ivfsoverlay %t/vfs.json -fsyntax-only %t/tu.m -verify

//--- Dir1/module.modulemap

//--- Dir2/module.private.modulemap
module Foo_Private {}

//--- vfs.json.in
{
  'version': 0,
  'use-external-names': true,
  'roots': [
    {
      'name': 'DIR/Virtual',
      'type': 'directory',
      'contents': [
        {
          'name': 'module.modulemap',
          'type': 'file',
          'external-contents': 'DIR/Dir1/module.modulemap'
        },
        {
          'name': 'module.private.modulemap',
          'type': 'file',
          'external-contents': 'DIR/Dir2/module.private.modulemap'
        }
      ]
    }
  ]
}

//--- tu.m
@import Foo_Private;
// expected-no-diagnostics
