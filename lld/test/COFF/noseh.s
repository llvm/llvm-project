# REQUIRES: x86
# RUN: llvm-mc -triple i686-w64-mingw32 %s -filetype=obj -o %t.obj
# RUN: lld-link -lldmingw %t.obj -out:%t.exe -entry:main
# RUN: llvm-readobj --file-headers %t.exe | FileCheck %s --check-prefix=DEFAULT
# RUN: lld-link -lldmingw %t.obj -out:%t.noseh.exe -entry:main -noseh
# RUN: llvm-readobj --file-headers %t.noseh.exe | FileCheck %s --check-prefix=NOSEH

# DEFAULT: Characteristics [
# DEFAULT-NOT:   IMAGE_DLL_CHARACTERISTICS_NO_SEH
# DEFAULT: ]

# NOSEH: Characteristics [
# NOSEH:   IMAGE_DLL_CHARACTERISTICS_NO_SEH
# NOSEH: ]

        .text
        .globl  _main
_main:
        ret
