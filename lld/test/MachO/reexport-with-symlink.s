# REQUIRES: aarch64
# UNSUPPORTED: system-windows
# RUN: rm -rf %t; split-file %s %t
# RUN: ln -s Versions/A/Developer %t/Developer/Library/Frameworks/Developer.framework/
# RUN: llvm-mc -filetype obj -triple arm64-apple-macos11.0 %t/test.s -o %t/test.o
# RUN: %lld -arch arm64 -platform_version macos 11.0 11.0 -o %t/test -framework Developer -F %t/Developer/Library/Frameworks -L %t/Developer/usr/lib %t/test.o -t | FileCheck %s

# CHECK: {{.*}}/Developer/Library/Frameworks/Developer.framework/Developer
# CHECK: {{.*}}/Developer/usr/lib/libDeveloperSupport.tbd(@rpath/libDeveloperSupport.dylib)
# CHECK-NOT: {{.*}}/Developer/Library/Frameworks/Developer.framework/Versions/A/Developer

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
        "name": "@rpath/Developer.framework/Developer"
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
#--- Developer/usr/lib/libDeveloperSupport.tbd
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
        "name": "@rpath/libDeveloperSupport.dylib"
      }
    ],
    "reexported_libraries": [
      {
        "names": [
          "@rpath/Developer.framework/Versions/A/Developer"
        ]
      }
    ],
    "exported_symbols": [
      {
        "text": {
          "global": ["_funcSupport"]
        }
      }
    ]
  }
}
#--- test.s
.text
.globl _main
.linker_option "-lDeveloperSupport"

_main:
  ret

.data
  .quad _funcPublic
  .quad _funcSupport
