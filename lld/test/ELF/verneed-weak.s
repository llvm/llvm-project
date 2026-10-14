# REQUIRES: x86
## Test that vna_flags is set to VER_FLG_WEAK if all references to a version
## are weak. This allows rtld to continue with a warning instead of erroring
## when the required version is not found.

# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 ref.s -o ref.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 refw.s -o refw.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 refw-gh.s -o refw-gh.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 def.s -o def.o
# RUN: ld.lld -shared --soname=def.so --version-script=ver def.o -o def.so

## All references are weak; vna_flags should be VER_FLG_WEAK.
# RUN: ld.lld -shared refw-gh.o def.so -o weakref.so
# RUN: llvm-readelf -V weakref.so | FileCheck %s --check-prefix=WEAK1

# WEAK1:      Version needs section '.gnu.version_r' contains 1 entries:
# WEAK1-NEXT:  Addr:
# WEAK1-NEXT:   0x0000: Version: 1  File: def.so  Cnt: 2
# WEAK1-NEXT:   0x0010:   Name: v1  Flags: WEAK  Version: 3
# WEAK1-NEXT:   0x0020:   Name: v2  Flags: WEAK  Version: 2

## All references are weak (from two object files); vna_flags should be VER_FLG_WEAK.
# RUN: ld.lld -shared refw.o refw-gh.o def.so -o weakref2.so
# RUN: llvm-readelf -V weakref2.so | FileCheck %s --check-prefix=WEAK2

# WEAK2:      Version needs section '.gnu.version_r' contains 1 entries:
# WEAK2-NEXT:  Addr:
# WEAK2-NEXT:   0x0000: Version: 1  File: def.so  Cnt: 2
# WEAK2-NEXT:   0x0010:   Name: v1  Flags: WEAK  Version: 2
# WEAK2-NEXT:   0x0020:   Name: v2  Flags: WEAK  Version: 3

## v1 has mixed references (none); v2 has weak references (WEAK).
# RUN: ld.lld -shared ref.o refw-gh.o def.so -o mixedref.so
# RUN: llvm-readelf -V mixedref.so | FileCheck %s --check-prefix=MIXED

# MIXED:      Version needs section '.gnu.version_r' contains 1 entries:
# MIXED-NEXT:  Addr:
# MIXED-NEXT:   0x0000: Version: 1  File: def.so  Cnt: 2
# MIXED-NEXT:   0x0010:   Name: v1  Flags: none  Version: 2
# MIXED-NEXT:   0x0020:   Name: v2  Flags: WEAK  Version: 3

## All references are non-weak; vna_flags should be 0 (none).
# RUN: ld.lld -shared ref.o def.so -o strongref.so
# RUN: llvm-readelf -V strongref.so | FileCheck %s --check-prefix=STRONG

# STRONG:      Version needs section '.gnu.version_r' contains 1 entries:
# STRONG-NEXT:  Addr:
# STRONG-NEXT:   0x0000: Version: 1  File: def.so  Cnt: 1
# STRONG-NEXT:   0x0010:   Name: v1  Flags: none  Version: 2
# STRONG-EMPTY:

## --as-needed: weak reference doesn't pull in def.so, so no version needs.
# RUN: ld.lld -shared refw.o --as-needed def.so -o asneeded.so
# RUN: llvm-readelf -V asneeded.so | count 0

#--- ver
v1 { f; g; };
v2 { h; };

#--- ref.s
.globl f
call f

#--- refw.s
.weak f
call f

#--- refw-gh.s
.weak g, h
.symver g, g@@@v1
call g
call h

#--- def.s
.globl f, g, h
f:
g:
h:
  ret
