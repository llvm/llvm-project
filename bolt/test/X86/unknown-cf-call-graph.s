## Check that indirect jumps with unknown control flow and set call profile are
## handled in call graph construction.

# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %t/src -o %t.o
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.out --lite=0 -print-cfg -data %t/yaml \
# RUN:   --print-only=main --reorder-functions=cdsort --dump-cg=%t.dot \
# RUN:   --profile-ignore-hash | FileCheck %s
# RUN: FileCheck --input-file %t.dot --check-prefix=CHECK-CG %s
# CHECK-CG:      digraph g {
# CHECK-CG-NEXT: f0 [label="main
# CHECK-CG-NEXT: f1 [label="foo
# CHECK-CG-NEXT: f0 -> f1 [label="normWgt=0.000,weight=1000,callOffset=0.0"];
# CHECK-CG-NEXT: }
#--- src
  .globl main
  .type main, %function
  .p2align 2
main:
   jmpq *%rax
# CHECK: jmpq *%rax # UNKNOWN CONTROL FLOW # CallProfile: 1000 (0 misses) :
# CHECK-NEXT:                              { foo: 1000 (0 misses) }
   jmpq *%rbx
.size main, .-main

  .globl foo
  .type foo, %function
  .p2align 2
foo:
   ud2
.size foo, .-foo
.reloc 0, R_X86_64_NONE
#--- yaml
---
header:
  profile-version: 1
  binary-name:     'test'
  binary-build-id: 0
  profile-flags:   [ lbr ]
  profile-origin:  perf data aggregator
  profile-events:  ''
  dfs-order:       false
  hash-func:       xxh3
functions:
  - name:    main
    fid:     1
    hash:    0x1
    exec:    1000
    nblocks: 1
    blocks:
      - bid:   0
        insns: 1
        hash:  0x1
        exec:  1000
        calls: [ { off: 0x0, fid: 2, cnt: 1000 } ]
  - name:    foo
    fid:     2
    hash:    0x2
    exec:    1000
    nblocks: 1
    blocks:
      - bid:   0
        insns: 1
        hash:  0x1
        exec:  1000
...
