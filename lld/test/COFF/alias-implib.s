# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64-mingw32 -o %t.o %s
# RUN: lld-link -lldmingw -entry:main -out:%t.exe %t.o %S/Inputs/alias-implib.lib -verbose 2>&1 | FileCheck %s

# Check that the undefined (weak) alias symbol (with an existing lazy
# __imp_alias) doesn't trigger trying to load __imp_alias for autoimport,
# when the weak alias target actually does exist.

# CHECK-NOT: Loading lazy{{.*}}for automatic import

    .text
    .globl main
main:
    call alias
    ret

# alias-implib.lib was created with "llvm-dlltool -m i386:x86-64
# -l alias-implib.lib -d alias-implib.def" with this def snippet:
# LIBRARY lib.dll
# EXPORTS
# realfunc
# alias == realfunc
