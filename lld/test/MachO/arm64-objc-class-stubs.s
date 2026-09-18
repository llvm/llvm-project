# REQUIRES: aarch64

# Tests arm64 ObjC class stub emission for fast/small stubs, local/dylib/archive
# classes, autolinked archives, dynamic lookup, malformed names, and missing
# classes.

# RUN: rm -rf %t && split-file %s %t

# Check fast and small stubs for regular selector calls, local class calls, and
# dylib class calls.
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/main.s -o %t/main.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/external.s -o %t/external.o
# RUN: %lld -arch arm64 -dylib -install_name @executable_path/libexternal.dylib \
# RUN:   -o %t/libexternal.dylib %t/external.o
# RUN: %lld -arch arm64 -lSystem -o %t/fast.out %t/main.o \
# RUN:   %t/libexternal.dylib -objc_stubs_fast
# RUN: llvm-objdump --no-show-raw-insn --section=__TEXT,__objc_stubs --syms \
# RUN:   --macho %t/fast.out | FileCheck %s --check-prefix=FAST
# RUN: %lld -arch arm64 -lSystem -o %t/small.out %t/main.o \
# RUN:   %t/libexternal.dylib -objc_stubs_small
# RUN: llvm-objdump --no-show-raw-insn --section=__TEXT,__objc_stubs --syms \
# RUN:   --macho %t/small.out | FileCheck %s --check-prefix=SMALL
# RUN: llvm-mc -filetype=obj -triple=arm64e-apple-darwin %t/main.s \
# RUN:   -o %t/main-arm64e.o
# RUN: llvm-mc -filetype=obj -triple=arm64e-apple-darwin %t/external.s \
# RUN:   -o %t/external-arm64e.o
# RUN: %lld -arch arm64e -dylib -install_name @executable_path/libexternal.dylib \
# RUN:   -o %t/libexternal-arm64e.dylib %t/external-arm64e.o
# RUN: %lld -arch arm64e -lSystem -o %t/fast-arm64e.out %t/main-arm64e.o \
# RUN:   %t/libexternal-arm64e.dylib -objc_stubs_fast
# RUN: llvm-objdump --no-show-raw-insn --section=__TEXT,__objc_stubs --syms \
# RUN:   --macho %t/fast-arm64e.out | FileCheck %s --check-prefix=FAST

# Check archive extraction through class-stub references.
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/archive-main.s \
# RUN:   -o %t/archive-main.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/archive-class.s \
# RUN:   -o %t/archive-class.o
# RUN: llvm-ar rcs %t/libarchive.a %t/archive-class.o
# RUN: %lld -arch arm64 -lSystem -o %t/archive.out %t/archive-main.o \
# RUN:   %t/libarchive.a -objc_stubs_fast -U _objc_msgSend
# RUN: llvm-objdump --no-show-raw-insn --section=__TEXT,__objc_stubs \
# RUN:   --macho %t/archive.out | FileCheck %s --check-prefix=ARCHIVE
# RUN: llvm-nm %t/archive.out | FileCheck %s --check-prefix=ARCHIVE-SYMS
# RUN: %lld -arch arm64 -lSystem -o %t/start-lib.out %t/archive-main.o \
# RUN:   --start-lib %t/archive-class.o --end-lib -objc_stubs_fast \
# RUN:   -U _objc_msgSend
# RUN: llvm-objdump --no-show-raw-insn --section=__TEXT,__objc_stubs \
# RUN:   --macho %t/start-lib.out | FileCheck %s --check-prefix=ARCHIVE
# RUN: llvm-nm %t/start-lib.out | FileCheck %s --check-prefix=ARCHIVE-SYMS

# Check autolink extraction when the archive member defines the class.
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/autolink-main-a.s \
# RUN:   -o %t/autolink-main-a.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/autolink-dep-a.s \
# RUN:   -o %t/autolink-dep-a.o
# RUN: llvm-ar rcs %t/libautolinka.a %t/autolink-dep-a.o
# RUN: %lld -arch arm64 -lSystem -o %t/autolink-a.out \
# RUN:   %t/autolink-main-a.o -L%t -objc_stubs_fast -U _objc_msgSend
# RUN: llvm-objdump --no-show-raw-insn --section=__TEXT,__objc_stubs \
# RUN:   --macho %t/autolink-a.out | FileCheck %s --check-prefix=AUTOLINK-A

# Check autolink extraction when the class is already known.
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/autolink-main-b.s \
# RUN:   -o %t/autolink-main-b.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/autolink-dep-b.s \
# RUN:   -o %t/autolink-dep-b.o
# RUN: llvm-ar rcs %t/libautolinkb.a %t/autolink-dep-b.o
# RUN: %lld -arch arm64 -lSystem -dead_strip -o %t/autolink-b.out \
# RUN:   %t/autolink-main-b.o -L%t -objc_stubs_fast -U _objc_msgSend
# RUN: llvm-objdump --no-show-raw-insn --section=__TEXT,__objc_stubs \
# RUN:   --macho %t/autolink-b.out | FileCheck %s --check-prefix=AUTOLINK-B
# RUN: llvm-nm %t/autolink-b.out | FileCheck %s --check-prefix=AUTOLINK-B-SYMS

# Check class stubs for dynamic class symbols.
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/dynamic.s \
# RUN:   -o %t/dynamic.o
# RUN: %lld -arch arm64 -lSystem -o %t/dynamic.out %t/dynamic.o \
# RUN:   -objc_stubs_fast -U '_OBJC_CLASS_$_DynamicClass' -U _objc_msgSend
# RUN: llvm-objdump --no-show-raw-insn --section=__TEXT,__objc_stubs \
# RUN:   --macho %t/dynamic.out | FileCheck %s --check-prefix=DYNAMIC
# RUN: %lld -arch arm64 -lSystem -o %t/dynamic-lookup.out %t/dynamic.o \
# RUN:   -objc_stubs_fast -undefined dynamic_lookup
# RUN: llvm-objdump --no-show-raw-insn --section=__TEXT,__objc_stubs \
# RUN:   --macho %t/dynamic-lookup.out | FileCheck %s --check-prefix=DYNAMIC

# Check diagnostics for missing class symbols.
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/missing.s \
# RUN:   -o %t/missing.o
# RUN: not %lld -arch arm64 -lSystem -o /dev/null %t/missing.o \
# RUN:   -objc_stubs_fast 2>&1 | FileCheck %s --check-prefix=MISSING

# Check diagnostics for malformed class-stub symbol names.
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/malformed.s \
# RUN:   -o %t/malformed.o
# RUN: not %lld -arch arm64 -lSystem -o /dev/null %t/malformed.o \
# RUN:   -objc_stubs_fast 2>&1 | FileCheck %s --check-prefix=MALFORMED

# FAST:      Contents of (__TEXT,__objc_stubs) section
# FAST:      _objc_msgSend$instance:
# FAST-NEXT: adrp    x1,
# FAST-NEXT: ldr     x1, {{.*}} ; Objc selector ref: instance
# FAST-NEXT: adrp    x16,
# FAST-NEXT: ldr     x16,
# FAST-NEXT: br      x16
# FAST-NEXT: brk     #0x1
# FAST-NEXT: brk     #0x1
# FAST-NEXT: brk     #0x1
# FAST-NEXT: _objc_msgSendClass$external$_OBJC_CLASS_$_ExternalClass:
# FAST-NEXT: adrp    x0,
# FAST-NEXT: ldr     x0, {{.*}} ; literal pool symbol address: _OBJC_CLASS_$_ExternalClass
# FAST-NEXT: adrp    x1,
# FAST-NEXT: ldr     x1, {{.*}} ; Objc selector ref: external
# FAST-NEXT: adrp    x16,
# FAST-NEXT: ldr     x16,
# FAST-NEXT: br      x16
# FAST-NEXT: brk     #0x1
# FAST-NEXT: _objc_msgSendClass$local$_OBJC_CLASS_$_LocalClass:
# FAST-NEXT: adrp    x0, [[#]] ; 0x[[#%x,FAST_LOCAL_CLASS_PAGE:]]
# FAST-NEXT: add     x0, x0, #0x[[#%x,FAST_LOCAL_CLASS_OFF:]]
# FAST-NEXT: adrp    x1,
# FAST-NEXT: ldr     x1, {{.*}} ; Objc selector ref: local
# FAST-NEXT: adrp    x16,
# FAST-NEXT: ldr     x16,
# FAST-NEXT: br      x16
# FAST-NEXT: brk     #0x1
# FAST-LABEL: SYMBOL TABLE:
# FAST-DAG: {{0*}}[[#%x,FAST_LOCAL_CLASS_PAGE+FAST_LOCAL_CLASS_OFF]] g     O __DATA,__data _OBJC_CLASS_$_LocalClass

# SMALL:      Contents of (__TEXT,__objc_stubs) section
# SMALL:      _objc_msgSend$instance:
# SMALL-NEXT: adrp    x1,
# SMALL-NEXT: ldr     x1, {{.*}} ; Objc selector ref: instance
# SMALL-NEXT: b       _objc_msgSend
# SMALL-NEXT: _objc_msgSendClass$external$_OBJC_CLASS_$_ExternalClass:
# SMALL-NEXT: adrp    x0,
# SMALL-NEXT: ldr     x0, {{.*}} ; literal pool symbol address: _OBJC_CLASS_$_ExternalClass
# SMALL-NEXT: adrp    x1,
# SMALL-NEXT: ldr     x1, {{.*}} ; Objc selector ref: external
# SMALL-NEXT: b       _objc_msgSend
# SMALL-NEXT: _objc_msgSendClass$local$_OBJC_CLASS_$_LocalClass:
# SMALL-NEXT: adrp    x0, [[#]] ; 0x[[#%x,SMALL_LOCAL_CLASS_PAGE:]]
# SMALL-NEXT: add     x0, x0, #0x[[#%x,SMALL_LOCAL_CLASS_OFF:]]
# SMALL-NEXT: adrp    x1,
# SMALL-NEXT: ldr     x1, {{.*}} ; Objc selector ref: local
# SMALL-NEXT: b       _objc_msgSend
# SMALL-LABEL: SYMBOL TABLE:
# SMALL-DAG: {{0*}}[[#%x,SMALL_LOCAL_CLASS_PAGE+SMALL_LOCAL_CLASS_OFF]] g     O __DATA,__data _OBJC_CLASS_$_LocalClass

# DYNAMIC:      Contents of (__TEXT,__objc_stubs) section
# DYNAMIC-NEXT: _objc_msgSendClass$dynamic$_OBJC_CLASS_$_DynamicClass:
# DYNAMIC-NEXT: adrp    x0,
# DYNAMIC-NEXT: ldr     x0, {{.*}} ; literal pool symbol address: _OBJC_CLASS_$_DynamicClass
# DYNAMIC-NEXT: adrp    x1,
# DYNAMIC-NEXT: ldr     x1, {{.*}} ; Objc selector ref: dynamic
# DYNAMIC-NEXT: adrp    x16,
# DYNAMIC-NEXT: ldr     x16, {{.*}} ; literal pool symbol address: _objc_msgSend
# DYNAMIC-NEXT: br      x16
# DYNAMIC-NEXT: brk     #0x1

# ARCHIVE:      Contents of (__TEXT,__objc_stubs) section
# ARCHIVE-NEXT: _objc_msgSendClass$archive$_OBJC_CLASS_$_ArchiveClass:
# ARCHIVE-NEXT: adrp    x0,
# ARCHIVE-NEXT: add     x0, x0,
# ARCHIVE-NEXT: adrp    x1,
# ARCHIVE-NEXT: ldr     x1, {{.*}} ; Objc selector ref: archive
# ARCHIVE-NEXT: adrp    x16,
# ARCHIVE-NEXT: ldr     x16, {{.*}} ; literal pool symbol address: _objc_msgSend
# ARCHIVE-NEXT: br      x16
# ARCHIVE-NEXT: brk     #0x1
#
# ARCHIVE-SYMS: _OBJC_CLASS_$_ArchiveClass

# AUTOLINK-A:      Contents of (__TEXT,__objc_stubs) section
# AUTOLINK-A-NEXT: _objc_msgSendClass$auto$_OBJC_CLASS_$_AutoClass:
# AUTOLINK-A-NEXT: adrp    x0,
# AUTOLINK-A-NEXT: add     x0, x0,
# AUTOLINK-A-NEXT: adrp    x1,
# AUTOLINK-A-NEXT: ldr     x1, {{.*}} ; Objc selector ref: auto
# AUTOLINK-A-NEXT: adrp    x16,
# AUTOLINK-A-NEXT: ldr     x16, {{.*}} ; literal pool symbol address: _objc_msgSend
# AUTOLINK-A-NEXT: br      x16
# AUTOLINK-A-NEXT: brk     #0x1
#
# AUTOLINK-B:      Contents of (__TEXT,__objc_stubs) section
# AUTOLINK-B-NEXT: _objc_msgSendClass$late$_OBJC_CLASS_$_LateClass:
# AUTOLINK-B-NEXT: adrp    x0,
# AUTOLINK-B-NEXT: add     x0, x0,
# AUTOLINK-B-NEXT: adrp    x1,
# AUTOLINK-B-NEXT: ldr     x1, {{.*}} ; Objc selector ref: late
# AUTOLINK-B-NEXT: adrp    x16,
# AUTOLINK-B-NEXT: ldr     x16, {{.*}} ; literal pool symbol address: _objc_msgSend
# AUTOLINK-B-NEXT: br      x16
# AUTOLINK-B-NEXT: brk     #0x1
#
# AUTOLINK-B-SYMS: _OBJC_CLASS_$_LateClass

# MISSING: error: undefined symbol: _OBJC_CLASS_$_MissingClass
# MISSING-NEXT: >>> referenced by objc class stub

# MALFORMED: error: malformed objc class stub symbol _objc_msgSendClass$malformed; expected _objc_msgSendClass$<selector>$_OBJC_CLASS_$_<class>
# MALFORMED-NOT: Objc selector ref:
# MALFORMED-NOT: undefined symbol: _objc_msgSendClass$malformed

#--- main.s
# Main fast/small case: one regular selector stub plus local and dylib class
# stubs.
.section __TEXT,__objc_methname,cstring_literals
Linstance:
  .asciz "instance"
Llocal:
  .asciz "local"
Lexternal:
  .asciz "external"

.section __DATA,__objc_selrefs,literal_pointers,no_dead_strip
.p2align 3
  .quad Linstance
  .quad Llocal
  .quad Lexternal

.text
.globl _objc_msgSend
_objc_msgSend:
  ret

.globl _main
_main:
  bl _objc_msgSend$instance
  bl _objc_msgSendClass$local$_OBJC_CLASS_$_LocalClass
  bl _objc_msgSendClass$external$_OBJC_CLASS_$_ExternalClass
  ret

.data
.globl _OBJC_CLASS_$_LocalClass
_OBJC_CLASS_$_LocalClass:
  .quad 0

.subsections_via_symbols

#--- external.s
# Provides the dylib-defined class used by main.s.
.data
.globl _OBJC_CLASS_$_ExternalClass
_OBJC_CLASS_$_ExternalClass:
  .quad 0

#--- dynamic.s
# Uses a class stub whose class symbol is resolved dynamically.
.section __TEXT,__objc_methname,cstring_literals
Ldynamic:
  .asciz "dynamic"

.section __DATA,__objc_selrefs,literal_pointers,no_dead_strip
.p2align 3
  .quad Ldynamic

.text
.globl _main
_main:
  bl _objc_msgSendClass$dynamic$_OBJC_CLASS_$_DynamicClass
  ret

#--- archive-main.s
# References a class stub whose class definition must be pulled from an archive.
.text
.globl _main
_main:
  bl _objc_msgSendClass$archive$_OBJC_CLASS_$_ArchiveClass
  ret

#--- archive-class.s
# Archive member defining the class used by archive-main.s.
.data
.globl _OBJC_CLASS_$_ArchiveClass
_OBJC_CLASS_$_ArchiveClass:
  .quad 0

#--- autolink-main-a.s
# Autolinks an archive member that contains both the stub user and class.
.linker_option "-lautolinka"
.text
.globl _main
_main:
  bl _dep_a
  ret

#--- autolink-dep-a.s
# Autolinked member containing the class-stub reference and class definition.
.text
.globl _dep_a
_dep_a:
  bl _objc_msgSendClass$auto$_OBJC_CLASS_$_AutoClass
  ret

.data
.globl _OBJC_CLASS_$_AutoClass
_OBJC_CLASS_$_AutoClass:
  .quad 0

#--- autolink-main-b.s
# Autolinks an archive member that references a class defined in this object.
.linker_option "-lautolinkb"
.text
.globl _main
_main:
  bl _dep_b
  ret

.data
.globl _OBJC_CLASS_$_LateClass
_OBJC_CLASS_$_LateClass:
  .quad 0

#--- autolink-dep-b.s
# Autolinked member that introduces a late class-stub reference.
.text
.globl _dep_b
_dep_b:
  bl _objc_msgSendClass$late$_OBJC_CLASS_$_LateClass
  ret

#--- missing.s
# References a live class stub whose class symbol is unresolved.
.section __TEXT,__objc_methname,cstring_literals
Lmissing:
  .asciz "missing"

.section __DATA,__objc_selrefs,literal_pointers,no_dead_strip
.p2align 3
  .quad Lmissing

.text
.globl _objc_msgSend
_objc_msgSend:
  ret

.globl _main
_main:
  bl _objc_msgSendClass$missing$_OBJC_CLASS_$_MissingClass
  ret

#--- malformed.s
# References a class stub symbol without the required class-symbol suffix.
.text
.globl _main
_main:
  bl _objc_msgSendClass$malformed
  ret
