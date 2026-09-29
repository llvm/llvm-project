// REQUIRES: x86
// RUN: split-file %s %t.dir && cd %t.dir

// RUN: llvm-mc -filetype=obj -triple=i686-windows test.s -o test.obj
// RUN: llvm-mc -filetype=obj -triple=i686-windows lib.s -o lib.obj
// RUN: lld-link -dll -noentry -out:out.dll test.obj -start-lib lib.obj -end-lib -lldmingw

#--- test.s
     .section .test,"dr"
     .rva _func@4

#--- lib.s
     .globl _func
_func:
     ret

     // These symbols don't have lazy entries in the symbol table initially,
     // but will be added during resolution from _func@4 to _func. Make sure this
     // scenario is handled properly.
     .weak_anti_dep _func@5
     .set _func@5,_func

     .weak_anti_dep _func@3
     .set _func@3,_func
