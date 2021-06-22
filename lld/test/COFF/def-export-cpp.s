# REQUIRES: x86
# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-windows-msvc %t/dllmain.s -o %t/dllmain.obj
# RUN: llvm-mc -filetype=obj -triple=x86_64-windows-msvc %t/path.s -o %t/path.obj
# RUN: lld-link -lib %t/dllmain.obj -out:%t/archive.lib
# RUN: echo -e "LIBRARY foo\nEXPORTS\n  GetPathOnDisk" > %t.def
# RUN: lld-link -entry:dllmain -dll -def:%t.def %t/path.obj %t/archive.lib -out:%t/foo.dll -implib:%t/foo.lib
# RUN: llvm-readobj %t/foo.lib | FileCheck -check-prefix IMPLIB %s
# RUN: llvm-readobj --coff-exports %t/foo.dll | FileCheck -check-prefix EXPORTS %s

# IMPLIB: File: foo.dll
# IMPLIB: Name type: undecorate
# IMPLIB-NEXT: Symbol: __imp_?GetPathOnDisk@@YA_NPEA_W@Z
# IMPLIB-NEXT: Symbol: ?GetPathOnDisk@@YA_NPEA_W@Z

# EXPORTS: Name: GetPathOnDisk

#--- path.s

        .def    "?GetPathOnDisk@@YA_NPEA_W@Z";
        .scl    2;
        .type   32;
        .endef
        .globl  "?GetPathOnDisk@@YA_NPEA_W@Z"
"?GetPathOnDisk@@YA_NPEA_W@Z":
        retq

#--- dllmain.s

        .def     dllmain;
        .scl    2;
        .type   32;
        .endef
        .globl  dllmain
dllmain:
        retq
