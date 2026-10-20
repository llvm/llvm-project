# REQUIRES: x86
# RUN: split-file %s %t.dir
# RUN: llvm-lib -machine:i386 -out:%t.dir/test32.lib -def:%t.dir/test32.def
# RUN: llvm-lib -machine:amd64 -out:%t.dir/test64.lib -def:%t.dir/test64.def
# RUN: llvm-mc -triple i686-windows-msvc %t.dir/test.s -filetype=obj -o %t.dir/test32.obj
# RUN: llvm-mc -triple x86_64-windows-msvc %t.dir/test.s -filetype=obj -o %t.dir/test64.obj

# RUN: not lld-link -dll -noentry -out:%t32.dll %t.dir/test32.obj %t.dir/test64.lib 2>&1 | FileCheck --check-prefix=ERR32 %s
# ERR32: error: test64.lib(test.dll): machine type x64 conflicts with x86

# RUN: not lld-link -dll -noentry -out:%t64.dll %t.dir/test64.obj %t.dir/test32.lib 2>&1 | FileCheck --check-prefix=ERR64 %s
# ERR64: error: test32.lib(test.dll): machine type x86 conflicts with x64

#--- test.s
        .def     @feat.00;
        .scl    3;
        .type   0;
        .endef
        .globl  @feat.00
@feat.00 = 1
        .data
        .rva __imp__test

#--- test32.def
NAME test.dll
EXPORTS
         test DATA

#--- test64.def
NAME test.dll
EXPORTS
         _test DATA
