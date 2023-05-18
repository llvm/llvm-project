# REQUIRES: x86

# RUN: echo '.data; .file "file2"; foo2:; .global bar2; .hidden bar2; bar2:' > %t2.s
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %t2.s -o %t2.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t1.o

# RUN: ld.lld -o %t %t1.o %t2.o --emit-relocs
# RUN: llvm-readelf --symbols --sections %t | FileCheck %s

## Check we sort local symbols to match the following order: 
## file1, local1, section1, hidden1, file2, local2, section2, hidden2 ...

# CHECK: Section Headers:
# CHECK:   [Nr] Name
# CHECK:   [ [[ST:.*]]] .text
# CHECK:   [ [[SD:.*]]] .data
# CHECK:   [ [[SC:.*]]] .comment

# CHECK:      Size Type    Bind   Vis      Ndx Name
# CHECK-NEXT:    0 NOTYPE  LOCAL  DEFAULT  UND
# CHECK-NEXT:    0 FILE    LOCAL  DEFAULT  ABS file1
# CHECK-NEXT:    0 NOTYPE  LOCAL  DEFAULT    1 foo1
# CHECK-NEXT:    0 SECTION LOCAL  DEFAULT    [[ST]]
# CHECK-NEXT:    0 NOTYPE  LOCAL  HIDDEN     1 bar1
# CHECK-NEXT:    0 FILE    LOCAL  DEFAULT  ABS file2
# CHECK-NEXT:    0 NOTYPE  LOCAL  DEFAULT    2 foo2
# CHECK-NEXT:    0 SECTION LOCAL  DEFAULT    [[SD]]
# CHECK-NEXT:    0 NOTYPE  LOCAL  HIDDEN     2 bar2
# CHECK-NEXT:    0 SECTION LOCAL  DEFAULT    [[SC]]

foo1:

.global bar1
.hidden bar1
bar1:

.file "file1"
