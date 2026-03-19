## If the original of a wrapped symbol becomes unreferenced after wrapping, it
## should be dropped from the dynamic symbol table even if defined in a shared
## library.

# REQUIRES: x86

# RUN: rm -rf %t && split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-elf %t/original.s -o %t/original.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-elf %t/wrapped.s -o %t/wrapped.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-elf %t/ref.s -o %t/ref.o
# RUN: ld.lld -shared -o %t/liboriginal.so -soname liboriginal.so %t/original.o
# RUN: ld.lld -shared -o %t/liboriginal-and-wrapped.so \
# RUN:   -soname liboriginal-and-wrapped.so %t/original.o %t/wrapped.o
# RUN: ld.lld -shared -o %t/libref-with-original.so %t/ref.o \
# RUN:   --as-needed %t/liboriginal.so --wrap foo
# RUN: llvm-readelf --dynamic --dyn-syms %t/libref-with-original.so | \
# RUN:   FileCheck --check-prefix=ORIGINAL %s
# RUN: ld.lld -shared -o %t/libref-with-original-and-wrapped.so %t/ref.o \
# RUN:   --as-needed %t/liboriginal-and-wrapped.so --wrap foo
# RUN: llvm-readelf --dynamic --dyn-syms %t/libref-with-original-and-wrapped.so | \
# RUN:   FileCheck --check-prefix=ORIGINAL-AND-WRAPPED %s

# ORIGINAL-NOT: (NEEDED) Shared library: [liboriginal.so]
# ORIGINAL:      Symbol table '.dynsym' contains 3 entries:
# ORIGINAL:      NOTYPE  LOCAL  DEFAULT   UND
# ORIGINAL-NEXT: NOTYPE  GLOBAL DEFAULT   UND __wrap_foo
# ORIGINAL-NEXT: NOTYPE  GLOBAL DEFAULT     6 ref

# ORIGINAL-AND-WRAPPED: (NEEDED) Shared library: [liboriginal-and-wrapped.so]
# ORIGINAL-AND-WRAPPED:      Symbol table '.dynsym' contains 3 entries:
# ORIGINAL-AND-WRAPPED:      NOTYPE  LOCAL  DEFAULT   UND
# ORIGINAL-AND-WRAPPED-NEXT: NOTYPE  GLOBAL DEFAULT   UND __wrap_foo
# ORIGINAL-AND-WRAPPED-NEXT: NOTYPE  GLOBAL DEFAULT     6 ref

#--- original.s
.globl foo
foo:
	retq

#--- wrapped.s
.globl __wrap_foo
__wrap_foo:
	retq

#--- ref.s
.globl ref
ref:
	jmp	foo@plt
