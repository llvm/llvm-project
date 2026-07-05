// This test checks that the OpenMP host IR file goes through VFS overlays.

// RUN: rm -rf %t
// RUN: split-file %s %t

// RUN: sed -e "s|DIR|%/t|g" %t/vfs.json.in > %t/vfs.json
// RUN: %clang_cc1 -fopenmp-simd -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm-bc %t/host.c -o %t/host.bc

// RUN: %clang_cc1 -fopenmp-simd -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %t/device.c -o - \
// RUN:   -fopenmp-is-target-device -fopenmp-host-ir-file-path %t/virtual/host.bc -ivfsoverlay %t/vfs.json -verify

//--- vfs.json.in
{
  'version': 0,
  'use-external-names': true,
  'roots': [
    {
      'name': 'DIR/virtual',
      'type': 'directory',
      'contents': [
        {
          'name': 'host.bc',
          'type': 'file',
          'external-contents': 'DIR/host.bc'
        }
      ]
    }
  ]
}

//--- host.c
//--- device.c
// expected-no-diagnostics
