# REQUIRES: x86
# RUN: rm -rf %t && split-file %s %t
# RUN: echo '.section .bar, "a"; .quad 1;' | \
# RUN:   llvm-mc -filetype=obj -triple=x86_64-pc-linux - -o %tfile1.o
# RUN: echo '.section .zed, "a"; .quad 2;' | \
# RUN:   llvm-mc -filetype=obj -triple=x86_64-pc-linux - -o %tfile2.o

## We have a file name and no input sections description. In that case, all
## sections from the file specified should be included. Check that.
# RUN: ld.lld -o %t/a -T %t/a.t %tfile1.o %tfile2.o
# RUN: llvm-objdump -s %t/a | FileCheck %s

# CHECK:      Contents of section .foo:
# CHECK-NEXT:  01000000 00000000 02000000 00000000

# RUN: not ld.lld -o /dev/null -T %t/b.t %tfile1.o 2>&1 | FileCheck %s --check-prefix=ERR
# RUN: not ld.lld -o /dev/null -T %t/c.t %tfile1.o 2>&1 | FileCheck %s --check-prefix=ERR
# RUN: not ld.lld -o /dev/null -T %t/d.t %tfile1.o 2>&1 | FileCheck %s --check-prefix=ERR

# ERR: error: {{.*}}.t:1: expected filename pattern

#--- a.t
SECTIONS {
 .foo : { *file1.o *file2.o }
}

#--- b.t
SECTIONS { .foo : { (*foo) } }

#--- c.t
SECTIONS { .foo : { (*(foo)) } }

#--- d.t
SECTIONS { .foo : { )(*foo) } }
