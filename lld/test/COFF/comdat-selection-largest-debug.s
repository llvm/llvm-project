# REQUIRES: x86

# Test IMAGE_COMDAT_SELECT_LARGEST with /debug. The image and the PDB must
# contain public symbols only from the final largest COMDAT group.
#
# /debug disables section GC by default. The /opt:ref variant verifies the
# same behavior when section GC is explicitly enabled.

# RUN: split-file %s %t.dir
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/small.s -o %t.small.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/large.s -o %t.large.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/root.s -o %t.root.obj

# RUN: lld-link /debug /opt:ref /pdb:%t.ref.pdb /entry:entry \
# RUN:   /subsystem:console /nodefaultlib \
# RUN:   %t.small.obj %t.large.obj %t.root.obj /out:%t.ref.exe
# RUN: llvm-pdbutil dump -publics %t.ref.pdb | \
# RUN:   FileCheck --check-prefix=PDB \
# RUN:     --implicit-check-not=small_public %s
# RUN: llvm-pdbutil dump -l %t.ref.pdb | \
# RUN:   FileCheck --check-prefix=LINES --implicit-check-not=small.cpp %s
# RUN: llvm-objdump -s %t.ref.exe | \
# RUN:   FileCheck --check-prefix=IMAGE \
# RUN:     --implicit-check-not=11111111 \
# RUN:     --implicit-check-not=22222222 %s

# RUN: lld-link /debug /pdb:%t.noref.pdb /entry:entry /subsystem:console \
# RUN:   /nodefaultlib %t.small.obj %t.large.obj %t.root.obj \
# RUN:   /out:%t.noref.exe
# RUN: llvm-pdbutil dump -publics %t.noref.pdb | \
# RUN:   FileCheck --check-prefix=PDB \
# RUN:     --implicit-check-not=small_public %s
# RUN: llvm-pdbutil dump -l %t.noref.pdb | \
# RUN:   FileCheck --check-prefix=LINES --implicit-check-not=small.cpp %s
# RUN: llvm-objdump -s %t.noref.exe | \
# RUN:   FileCheck --check-prefix=IMAGE \
# RUN:     --implicit-check-not=11111111 \
# RUN:     --implicit-check-not=22222222 %s

# The 12-byte candidate must prevail. The small candidate's 0x11111111 and
# 0x22222222 markers must not occur anywhere in the output image.

# IMAGE: Contents of section .text:
# IMAGE-NEXT:  140001000 44444444 55555555 66666666

# The public stream must contain symbols backed by the prevailing chunk.
# small_public belongs exclusively to the discarded candidate and is excluded
# through --implicit-check-not above.

# PDB: Public Symbols
# PDB-DAG: S_PUB32 {{.*}} `leader`
# PDB-DAG: S_PUB32 {{.*}} `large_public`
# PDB-DAG: S_PUB32 {{.*}} `entry`

# The line table comes from a real associative .debug$S chunk. The discarded
# candidate's file is excluded through --implicit-check-not above.

# LINES: large.cpp

#--- small.s
        .cv_file 1 "small.cpp" "11111111111111111111111111111111" 1

        .section .text$largest, "xr", largest, leader

        .globl leader
leader:
        .cv_func_id 0
        .cv_loc 0 1 1 0 is_stmt 0
        .long 0x11111111

        .globl small_public
small_public:
        .long 0x22222222
.Lsmall_end:

        .section .debug$S, "dr", associative, leader
        .long 4
        .cv_linetable 0, leader, .Lsmall_end

        .section .debug$S, "dr"
        .long 4
        .cv_filechecksums
        .cv_stringtable

#--- large.s
        .cv_file 1 "large.cpp" "44444444444444444444444444444444" 1

        .section .text$largest, "xr", largest, leader

        .globl leader
leader:
        .cv_func_id 0
        .cv_loc 0 1 1 0 is_stmt 0
        .long 0x44444444

        .globl large_public
large_public:
        .long 0x55555555

        .long 0x66666666
.Llarge_end:

        .section .debug$S, "dr", associative, leader
        .long 4
        .cv_linetable 0, leader, .Llarge_end

        .section .debug$S, "dr"
        .long 4
        .cv_filechecksums
        .cv_stringtable

#--- root.s
        .section .text$root, "xr"

        .globl entry
entry:
        leaq leader(%rip), %rax
        retq
