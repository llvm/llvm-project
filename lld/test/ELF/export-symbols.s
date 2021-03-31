# REQUIRES: x86
## Verify that the arguments --export-dynamic and --dynamic-list
## put the correct symbols in the dynamic symbol table.

# RUN: echo "{ *; };" > %t.list

# RUN: echo ".globl shared" > %t.s ; echo "shared = 0xDEADBEEF" >> %t.s

# RUN: llvm-mc -filetype=obj -triple=x86_64 %t.s -o %t-shared.o
# RUN: ld.lld --shared %t-shared.o -o %t.so

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
## Use --fatal-warnings to confirm no diagnostics are emitted.
# RUN: ld.lld --fatal-warnings --defsym=defsym=_start %t.so %t.o -o %t.out
# RUN: ld.lld --fatal-warnings --defsym=defsym=_start %t.so %t.o -o %texport.out --export-dynamic
# RUN: ld.lld --fatal-warnings --defsym=defsym=_start %t.so %t.o -o %tlist.out --dynamic-list %t.list

# RUN: llvm-readelf --dyn-syms %t.out | FileCheck %s --check-prefix=NO-EXPORT
# RUN: llvm-readelf --dyn-syms %texport.out | FileCheck %s --check-prefix=EXPORT
# RUN: llvm-readelf --dyn-syms %tlist.out | FileCheck %s --check-prefix=EXPORT

# NO-EXPORT:      Symbol table '.dynsym' contains 3 entries:
# NO-EXPORT:      GLOBAL DEFAULT {{.*}} shared
# NO-EXPORT-NEXT: WEAK   DEFAULT {{.*}} undef_weak

# EXPORT:      Symbol table '.dynsym' contains 8 entries:
# EXPORT:      GLOBAL DEFAULT   {{.*}} shared
# EXPORT-NEXT: WEAK   DEFAULT   {{.*}} undef_weak
# EXPORT-NEXT: WEAK   DEFAULT   {{.*}} weak_default
# EXPORT-NEXT: GLOBAL DEFAULT   {{.*}} common
# EXPORT-NEXT: GLOBAL DEFAULT   ABS    abs
# EXPORT-NEXT: GLOBAL PROTECTED {{.*}} _start
# EXPORT-NEXT: GLOBAL DEFAULT   {{.*}} defsym

.weak undef_weak

.weak weak_default
weak_default:

.weak weak_internal
.internal weak_internal
weak_internal:

.weak weak_hidden
.internal weak_hidden
weak_hidden:

.weak weak_protected
.internal weak_protected
weak_protected:

.globl shared

.local local
local:

.comm common, 10

.globl abs
abs = 0xDEADBEEF

.globl hidden
.hidden hidden
hidden:

.globl _start
.protected _start
_start:
