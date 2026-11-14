// REQUIRES: x86-registered-target

// Test that -sample-profile-supplement and
// -sample-profile-supplement-remapping-file read through LLVMContext's VFS
// (and thus comply with Clang's IO sandbox and respect -ivfsoverlay).

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s@DIR@%{/t:regex_replacement}@g" %t/vfs.yaml.in > %t/vfs.yaml

// Without a supplement, block #1 (`baz()`) has 0 Propeller samples and is
// placed in `.text.split._Z3foob`.
// RUN: %clang_cc1 -triple x86_64-pc-linux -S -O0 -debug-info-kind=line-tables-only \
// RUN:   -ffunction-sections -fbasic-block-sections=list=%t/propeller.txt \
// RUN:   -mllvm -basic-block-section-match-infer -mllvm -propeller-match-bbid \
// RUN:   %t/test.cpp -o - | FileCheck %s --check-prefixes=CHECK,WITHOUT-SUPPLEMENT

// With a supplement and remapping file mapped only through the VFS overlay,
// `_Z3barb` remaps to `_Z3foob`, block #1's weight scales to 1, and `baz()`
// stays in `.text._Z3foob`.
// RUN: %clang_cc1 -triple x86_64-pc-linux -S -O0 -debug-info-kind=line-tables-only \
// RUN:   -ivfsoverlay %t/vfs.yaml \
// RUN:   -ffunction-sections -fbasic-block-sections=list=%t/propeller.txt \
// RUN:   -mllvm -basic-block-section-match-infer -mllvm -propeller-match-bbid \
// RUN:   -mllvm -sample-profile-supplement=%/t/virtual.prof \
// RUN:   -mllvm -sample-profile-supplement-remapping-file=%/t/virtual.remap \
// RUN:   %t/test.cpp -o - | FileCheck %s --check-prefixes=CHECK,WITH-SUPPLEMENT

// CHECK:                    .section        .text._Z3foob,"ax",@progbits
// CHECK-LABEL:              _Z3foob:

// WITHOUT-SUPPLEMENT:       .section        .text.split._Z3foob,"ax",@progbits
// WITHOUT-SUPPLEMENT-LABEL: _Z3foob.cold:
// WITHOUT-SUPPLEMENT:       callq   _Z3bazv

// WITH-SUPPLEMENT:          callq   _Z3bazv
// WITH-SUPPLEMENT-NOT:      .section        .text.split._Z3foob

//--- test.cpp
void baz();

void foo(bool cond) {
  if (!cond)
    baz();
}

//--- propeller.txt
v1
f _Z3foob
g 0:100,1:0,2:100 1:0,2:0 2:100

//--- real.prof
_Z3barb:2000:1000
 2: 5

//--- real.remap
name 3foo 3bar

//--- vfs.yaml.in
{
  "version": 0,
  "roots": [
    {
      "name": "DIR",
      "type": "directory",
      "contents": [
        {
          "name": "virtual.prof",
          "type": "file",
          "external-contents": "DIR/real.prof"
        },
        {
          "name": "virtual.remap",
          "type": "file",
          "external-contents": "DIR/real.remap"
        }
      ]
    }
  ]
}
