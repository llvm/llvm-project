// This test checks that VFS usage doesn't leak between modules.

// RUN: rm -rf %t && split-file %s %t
// RUN: sed -e "s|DIR|%/t|g" %t/build/cdb.json.in > %t/build/cdb.json
// RUN: sed -e "s|DIR|%/t|g" %t/build/vfs.yaml.in > %t/build/vfs.yaml
// RUN: clang-scan-deps -compilation-database %t/build/cdb.json \
// RUN:   -format experimental-full --optimize-args=vfs > %t/deps.json
// RUN: cat %t/deps.json | sed 's:\\\\\?:/:g' | FileCheck %s -DPREFIX=%/t

// CHECK:      {
// CHECK-NEXT:   "modules": [
// CHECK-NEXT:     {
// CHECK-NEXT:       "clang-module-deps": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "context-hash": "{{.*}}",
// CHECK-NEXT:           "module-name": "B"
// CHECK-NEXT:         },
// CHECK-NEXT:         {
// CHECK-NEXT:           "context-hash": "{{.*}}",
// CHECK-NEXT:           "module-name": "C"
// CHECK-NEXT:         }
// CHECK-NEXT:       ],
// CHECK-NEXT:       "clang-modulemap-file": "[[PREFIX]]/moduleA/module.modulemap",
// CHECK-NEXT:       "command-line": [
// Module A needs the VFS overlay because its dependency, module B, needs it.
// CHECK:              "-ivfsoverlay"
// CHECK-NEXT:         "[[PREFIX]]/build/vfs.yaml"
// CHECK:            ],
// CHECK-NEXT:       "context-hash": "{{.*}}",
// CHECK-NEXT:       "file-deps": [
// CHECK:            ],
// CHECK-NEXT:       "link-libraries": [],
// CHECK-NEXT:       "name": "A"
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:       "clang-module-deps": [],
// CHECK-NEXT:       "clang-modulemap-file": "[[PREFIX]]/moduleB/module.modulemap",
// CHECK-NEXT:       "command-line": [
// Module B needs the VFS overlay because it provides the header referred to by the module map.
// CHECK:              "-ivfsoverlay"
// CHECK-NEXT:         "[[PREFIX]]/build/vfs.yaml"
// CHECK:            ],
// CHECK-NEXT:       "context-hash": "{{.*}}",
// CHECK-NEXT:       "file-deps": [
// CHECK:            ],
// CHECK-NEXT:       "link-libraries": [],
// CHECK-NEXT:       "name": "B"
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:       "clang-module-deps": [],
// CHECK-NEXT:       "clang-modulemap-file": "[[PREFIX]]/moduleC/module.modulemap",
// CHECK-NEXT:       "command-line": [
// Module C doesn't need the VFS overlay.
// CHECK-NOT:          "-ivfsoverlay"
// CHECK:            ],
// CHECK-NEXT:       "context-hash": "{{.*}}",
// CHECK-NEXT:       "file-deps": [
// CHECK:            ],
// CHECK-NEXT:       "link-libraries": [],
// CHECK-NEXT:       "name": "C"
// CHECK-NEXT:     }
// CHECK-NEXT:   ],
// CHECK-NEXT:   "translation-units": [
// CHECK:        ]
// CHECK:      }

//--- build/cdb.json.in
[{
  "directory": "DIR",
  "command": "clang -c DIR/tu.m -I DIR/moduleA -I DIR/moduleB -I DIR/moduleC -fmodules -fmodules-cache-path=DIR/cache -fimplicit-module-maps -ivfsoverlay DIR/build/vfs.yaml",
  "file": "DIR/tu.m"
}]

//--- build/vfs.yaml.in
{
  "version": 0,
  "case-sensitive": "false",
  "roots": [
    {
      "contents": [
        {
          "external-contents": "DIR/build/B.h",
          "name": "B.h",
          "type": "file"
        }
      ],
      "name": "DIR/moduleB",
      "type": "directory"
    }
  ]
}

//--- tu.m
@import A;

//--- moduleA/module.modulemap
module A { header "A.h" }
//--- moduleA/A.h
@import B;
@import C;

//--- moduleB/module.modulemap
module B { header "B.h" }
//--- build/B.h

//--- moduleC/module.modulemap
module C { header "C.h" }
//--- moduleC/C.h
