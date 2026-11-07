# REQUIRES: x86

# Test IMAGE_COMDAT_SELECT_LARGEST with multiple associative sections.
# Only the final largest leader and all of its associative sections must be
# emitted, independent of input order and section-GC configuration.

# RUN: sed -e s/TYPE/.byte/ -e s/SIZE/1/ %s | \
# RUN:   llvm-mc -triple x86_64-pc-win32 -filetype=obj -o %t.1.obj
# RUN: sed -e s/TYPE/.short/ -e s/SIZE/2/ %s | \
# RUN:   llvm-mc -triple x86_64-pc-win32 -filetype=obj -o %t.2.obj
# RUN: sed -e s/TYPE/.long/ -e s/SIZE/4/ %s | \
# RUN:   llvm-mc -triple x86_64-pc-win32 -filetype=obj -o %t.4.obj

        .section .text$aa, "", associative, symbol
        .byte SIZE, 0xaa, 0xaa, 0xaa

        .section .text$ab, "", associative, symbol
        .byte SIZE, 0xbb, 0xbb, 0xbb

        .section .text$nm, "", largest, symbol
        .globl symbol
symbol:
        TYPE SIZE

# Exercise successive replacements, a losing candidate before a replacement,
# and the largest candidate arriving first. The default is /opt:ref, so use the
# two GC modes explicitly instead of testing the default separately.

# RUN: lld-link /opt:ref /include:symbol /dll /noentry /nodefaultlib \
# RUN:   %t.1.obj %t.2.obj %t.4.obj /out:%t.ref.124.exe
# RUN: llvm-objdump -s %t.ref.124.exe | FileCheck --check-prefix=LARGEST4 %s
# RUN: lld-link /opt:noref /include:symbol /dll /noentry /nodefaultlib \
# RUN:   %t.1.obj %t.2.obj %t.4.obj /out:%t.noref.124.exe
# RUN: llvm-objdump -s %t.noref.124.exe | FileCheck --check-prefix=LARGEST4 %s

# RUN: lld-link /opt:ref /include:symbol /dll /noentry /nodefaultlib \
# RUN:   %t.2.obj %t.1.obj %t.4.obj /out:%t.ref.214.exe
# RUN: llvm-objdump -s %t.ref.214.exe | FileCheck --check-prefix=LARGEST4 %s
# RUN: lld-link /opt:noref /include:symbol /dll /noentry /nodefaultlib \
# RUN:   %t.2.obj %t.1.obj %t.4.obj /out:%t.noref.214.exe
# RUN: llvm-objdump -s %t.noref.214.exe | FileCheck --check-prefix=LARGEST4 %s

# RUN: lld-link /opt:ref /include:symbol /dll /noentry /nodefaultlib \
# RUN:   %t.4.obj %t.2.obj %t.1.obj /out:%t.ref.421.exe
# RUN: llvm-objdump -s %t.ref.421.exe | FileCheck --check-prefix=LARGEST4 %s
# RUN: lld-link /opt:noref /include:symbol /dll /noentry /nodefaultlib \
# RUN:   %t.4.obj %t.2.obj %t.1.obj /out:%t.noref.421.exe
# RUN: llvm-objdump -s %t.noref.421.exe | FileCheck --check-prefix=LARGEST4 %s

# LARGEST4: Contents of section .text:
# LARGEST4-NEXT:  180001000 04aaaaaa 04bbbbbb 04000000
