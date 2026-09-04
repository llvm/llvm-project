# REQUIRES: aarch64
# Windows does not support rpath
# UNSUPPORTED: system-windows
# RUN: rm -rf %t; split-file %s %t
# RUN: ln -s Versions/A/Developer %t/Developer/Library/Frameworks/Developer.framework/
# RUN: ln -s Versions/A/DeveloperCore %t/Developer/Library/PrivateFrameworks/DeveloperCore.framework/
# RUN: llvm-mc -filetype obj -triple arm64-apple-macos11.0 %t/test.s -o %t/test.o
# RUN: %lld -arch arm64 -platform_version macos 11.0 11.0 -o %t/test -framework Developer -F %t/Developer/Library/Frameworks %t/test.o

# RUN: llvm-objdump --bind --no-show-raw-insn -d %t/test | FileCheck %s
# CHECK:     Bind table:
# CHECK-DAG: __DATA __data {{.*}} pointer 0 Developer         _funcPublic
# CHECK-DAG: __DATA __data {{.*}} pointer 0 Developer         _funcCore

#--- Developer/Library/Frameworks/Developer.framework/Versions/A/Developer
{
  "tapi_tbd_version": 5,
  "main_library": {
    "target_info": [
      {
        "target": "arm64-macos"
      }
    ],
    "install_names": [
      {
        "name": "@rpath/Developer.framework/Versions/A/Developer"
      }
    ],
    "rpaths": [
      {
        "paths": [
          "@loader_path/../../../../PrivateFrameworks/"
        ]
      }
    ],
    "reexported_libraries": [
      {
        "names": [
          "@rpath/DeveloperCore.framework/Versions/A/DeveloperCore"
        ]
      }
    ],
    "exported_symbols": [
      {
        "text": {
          "global": ["_funcPublic"]
        }
      }
    ]
  }
}
#--- Developer/Library/PrivateFrameworks/DeveloperCore.framework/Versions/A/DeveloperCore
{
  "tapi_tbd_version": 5,
  "main_library": {
    "target_info": [
      {
        "target": "arm64-macos"
      }
    ],
    "install_names": [
      {
        "name": "@rpath/DeveloperCore.framework/Versions/A/DeveloperCore"
      }
    ],
    "allowable_clients": [
      {
        "clients": ["Developer"]
      }
    ],
    "exported_symbols": [
      {
        "text": {
          "global": ["_funcCore"]
        }
      }
    ]
  }
}
#--- test.s
.text
.globl _main

_main:
  ret

.data
  .quad _funcPublic
  .quad _funcCore
