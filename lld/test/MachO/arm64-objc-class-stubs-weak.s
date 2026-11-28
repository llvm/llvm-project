# REQUIRES: aarch64

# Class stubs must preserve the weak import of their retained ordinary classref.
# This is the representation emitted by Clang for weakly imported class receivers.
# Keep real strong references and class stubs with no weak classref strong.

# RUN: rm -rf %t && split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/weak.s -o %t/weak.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/strong.s -o %t/strong.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/bare.s -o %t/bare.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/weak-ref-only.s -o %t/weak-ref-only.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/dylib.s -o %t/dylib.o
# RUN: %lld -arch arm64 -dylib -install_name @rpath/libFuture.dylib \
# RUN:   -o %t/libFuture.dylib %t/dylib.o

# Bitcode supplies the weak class symbol and a retained ordinary classref.
# A module summary exercises ThinLTO, as in production class-stub builds.
# RUN: llvm-as %t/weak.ll -o %t/weak.bc
# RUN: opt -module-summary %t/weak.bc -o %t/weak.thin.bc
# RUN: %lld -arch arm64 -dead_strip -fixup_chains -objc_stubs_fast \
# RUN:   -o %t/weak-thin-fast.out %t/weak.thin.bc %t/libFuture.dylib
# RUN: llvm-nm -m %t/weak-thin-fast.out | FileCheck %s --check-prefix=WEAK-NM
# RUN: llvm-objdump --macho --chained-fixups %t/weak-thin-fast.out | FileCheck %s --check-prefix=WEAK-CHAIN
# RUN: llvm-objdump --macho --no-show-raw-insn --section=__TEXT,__objc_stubs %t/weak-thin-fast.out | FileCheck %s --check-prefix=STUB
# RUN: %lld -arch arm64 -dead_strip -fixup_chains -objc_stubs_small \
# RUN:   -o %t/weak-thin-small.out %t/weak.thin.bc %t/libFuture.dylib
# RUN: llvm-nm -m %t/weak-thin-small.out | FileCheck %s --check-prefix=WEAK-NM
# RUN: llvm-objdump --macho --chained-fixups %t/weak-thin-small.out | FileCheck %s --check-prefix=WEAK-CHAIN
# RUN: llvm-objdump --macho --no-show-raw-insn --section=__TEXT,__objc_stubs %t/weak-thin-small.out | FileCheck %s --check-prefix=STUB

# Check both stub modes and serialized chained import flags with dead stripping.
# RUN: %lld -arch arm64 -dead_strip -dead_strip_dylibs -fixup_chains \
# RUN:   -objc_stubs_fast -o %t/weak-fast.out %t/weak.o %t/libFuture.dylib
# RUN: llvm-nm -m %t/weak-fast.out | FileCheck %s --check-prefix=WEAK-NM
# RUN: llvm-objdump --macho --chained-fixups %t/weak-fast.out | FileCheck %s --check-prefix=WEAK-CHAIN
# RUN: llvm-objdump --macho --no-show-raw-insn --section=__TEXT,__objc_stubs %t/weak-fast.out | FileCheck %s --check-prefix=STUB
# RUN: %lld -arch arm64 -dead_strip -dead_strip_dylibs -fixup_chains \
# RUN:   -objc_stubs_small -o %t/weak-small.out %t/weak.o %t/libFuture.dylib
# RUN: llvm-nm -m %t/weak-small.out | FileCheck %s --check-prefix=WEAK-NM
# RUN: llvm-objdump --macho --chained-fixups %t/weak-small.out | FileCheck %s --check-prefix=WEAK-CHAIN
# RUN: llvm-objdump --macho --no-show-raw-insn --section=__TEXT,__objc_stubs %t/weak-small.out | FileCheck %s --check-prefix=STUB

# The GOT and ordinary classref must both keep weak_import in legacy bind info.
# RUN: %lld -arch arm64 -no_fixup_chains -objc_stubs_fast \
# RUN:   -o %t/weak-fast-bind.out %t/weak.o %t/libFuture.dylib
# RUN: llvm-objdump --macho --bind %t/weak-fast-bind.out | FileCheck %s --check-prefix=WEAK-BIND
# RUN: %lld -arch arm64 -dead_strip -no_fixup_chains -objc_stubs_small \
# RUN:   -o %t/weak-small-bind.out %t/weak.o %t/libFuture.dylib
# RUN: llvm-objdump --macho --bind %t/weak-small-bind.out | FileCheck %s --check-prefix=WEAK-BIND

# The retained weak classref may be provided by a separate object. Match the
# same weak import behavior in either order, even when the stub is bare.
# RUN: %lld -arch arm64 -dead_strip -fixup_chains -objc_stubs_fast \
# RUN:   -o %t/bare-weak.out %t/bare.o %t/weak-ref-only.o %t/libFuture.dylib
# RUN: llvm-nm -m %t/bare-weak.out | FileCheck %s --check-prefix=WEAK-NM
# RUN: llvm-objdump --macho --chained-fixups %t/bare-weak.out | FileCheck %s --check-prefix=WEAK-CHAIN
# RUN: %lld -arch arm64 -dead_strip -fixup_chains -objc_stubs_small \
# RUN:   -o %t/weak-bare.out %t/weak-ref-only.o %t/bare.o %t/libFuture.dylib
# RUN: llvm-nm -m %t/weak-bare.out | FileCheck %s --check-prefix=WEAK-NM
# RUN: llvm-objdump --macho --chained-fixups %t/weak-bare.out | FileCheck %s --check-prefix=WEAK-CHAIN

# A real strong reference dominates weak references in either object-file order.
# RUN: %lld -arch arm64 -dead_strip -fixup_chains -objc_stubs_fast \
# RUN:   -o %t/weak-strong.out %t/weak.o %t/strong.o %t/libFuture.dylib
# RUN: llvm-nm -m %t/weak-strong.out | FileCheck %s --check-prefix=STRONG-NM
# RUN: llvm-objdump --macho --chained-fixups %t/weak-strong.out | FileCheck %s --check-prefix=STRONG-CHAIN
# RUN: %lld -arch arm64 -dead_strip -fixup_chains -objc_stubs_small \
# RUN:   -o %t/strong-weak.out %t/strong.o %t/weak.o %t/libFuture.dylib
# RUN: llvm-nm -m %t/strong-weak.out | FileCheck %s --check-prefix=STRONG-NM
# RUN: llvm-objdump --macho --chained-fixups %t/strong-weak.out | FileCheck %s --check-prefix=STRONG-CHAIN

# A bare class stub has no weak metadata, so the class remains a strong import.
# RUN: %lld -arch arm64 -dead_strip -fixup_chains -objc_stubs_fast \
# RUN:   -o %t/bare.out %t/bare.o %t/libFuture.dylib
# RUN: llvm-nm -m %t/bare.out | FileCheck %s --check-prefix=STRONG-NM
# RUN: llvm-objdump --macho --chained-fixups %t/bare.out | FileCheck %s --check-prefix=STRONG-CHAIN

# An explicitly weak dylib still controls effective import weakness, including
# symbols also referenced strongly by an ordinary object.
# RUN: %lld -arch arm64 -dead_strip -fixup_chains -objc_stubs_fast \
# RUN:   -o %t/weak-library-bare.out %t/bare.o -weak_library %t/libFuture.dylib
# RUN: llvm-nm -m %t/weak-library-bare.out | FileCheck %s --check-prefix=WEAK-NM
# RUN: llvm-objdump --macho --chained-fixups %t/weak-library-bare.out | FileCheck %s --check-prefix=WEAK-CHAIN
# RUN: %lld -arch arm64 -dead_strip -fixup_chains -objc_stubs_small \
# RUN:   -o %t/weak-library-strong.out %t/weak.o %t/strong.o -weak_library %t/libFuture.dylib
# RUN: llvm-nm -m %t/weak-library-strong.out | FileCheck %s --check-prefix=WEAK-NM
# RUN: llvm-objdump --macho --chained-fixups %t/weak-library-strong.out | FileCheck %s --check-prefix=WEAK-CHAIN

# WEAK-NM: (undefined) weak external _OBJC_CLASS_$_FutureClass (from libFuture)
# STRONG-NM: (undefined) external _OBJC_CLASS_$_FutureClass (from libFuture)

# WEAK-CHAIN: weak_import = 1{{[[:space:]]+}}name_offset = {{[0-9]+}} (_OBJC_CLASS_$_FutureClass)
# STRONG-CHAIN: weak_import = 0{{[[:space:]]+}}name_offset = {{[0-9]+}} (_OBJC_CLASS_$_FutureClass)

# WEAK-BIND-DAG: __got {{.*}} _OBJC_CLASS_$_FutureClass (weak_import)
# WEAK-BIND-DAG: __objc_classrefs {{.*}} _OBJC_CLASS_$_FutureClass (weak_import)

# STUB: _objc_msgSendClass$factory$_OBJC_CLASS_$_FutureClass:
# STUB-NEXT: adrp x0,
# STUB-NEXT: ldr x0,

#--- weak.s
.text
.globl _main, _objc_msgSend
.no_dead_strip _objc_msgSend
_main:
  bl _objc_msgSendClass$factory$_OBJC_CLASS_$_FutureClass
  mov w0, #0
  ret
_objc_msgSend:
  ret

.weak_reference _OBJC_CLASS_$_FutureClass
.section __DATA,__objc_classrefs,regular,no_dead_strip
.p2align 3
Lclassref:
  .quad _OBJC_CLASS_$_FutureClass
.subsections_via_symbols

#--- strong.s
.section __DATA,__data
.globl _strong_reference
.no_dead_strip _strong_reference
_strong_reference:
  .quad _OBJC_CLASS_$_FutureClass
.subsections_via_symbols

#--- weak-ref-only.s
.weak_reference _OBJC_CLASS_$_FutureClass
.section __DATA,__objc_classrefs,regular,no_dead_strip
.p2align 3
Lclassref:
  .quad _OBJC_CLASS_$_FutureClass
.subsections_via_symbols

#--- bare.s
.text
.globl _main, _objc_msgSend
.no_dead_strip _objc_msgSend
_main:
  bl _objc_msgSendClass$factory$_OBJC_CLASS_$_FutureClass
  mov w0, #0
  ret
_objc_msgSend:
  ret
.subsections_via_symbols

#--- dylib.s
.data
.globl _OBJC_CLASS_$_FutureClass
_OBJC_CLASS_$_FutureClass:
  .quad 0

.text
.p2align 2
.globl _objc_msgSend
_objc_msgSend:
  ret

#--- weak.ll
target triple = "arm64-apple-macosx11.0.0"

@"OBJC_CLASS_$_FutureClass" = extern_weak global i8
@"OBJC_CLASSLIST_REFERENCES_$_" = internal global ptr @"OBJC_CLASS_$_FutureClass", section "__DATA,__objc_classrefs,regular,no_dead_strip", align 8
@llvm.compiler.used = appending global [1 x ptr] [ptr @"OBJC_CLASSLIST_REFERENCES_$_"], section "llvm.metadata"

define i32 @main() {
  %result = call ptr @"objc_msgSendClass$factory$_OBJC_CLASS_$_FutureClass"(ptr poison, ptr undef)
  ret i32 0
}

declare ptr @"objc_msgSendClass$factory$_OBJC_CLASS_$_FutureClass"(ptr, ptr, ...)
