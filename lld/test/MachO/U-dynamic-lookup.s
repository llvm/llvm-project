# REQUIRES: x86
# RUN: rm -rf %t
# RUN: split-file %s %t

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos -o %t/foo.o %t/foo.s
# RUN: %lld -dylib -o %t/foo.dylib %t/foo.o

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos -o %t/main.o %t/main.s

# _foo starts out as a (non-weak) dynamically looked up symbol and is merged
# against the Undefined from foo.o. _bar isn't referenced in any object file,
# but starts out as Undefined because of the -u flag. _baz isn't referenced
# at all.
# RUN: %lld -lSystem %t/main.o -U _foo -U _bar -u _bar -U _baz -o %t/out
# RUN: llvm-objdump --macho --lazy-bind %t/out | FileCheck --check-prefix=DYNAMIC %s
# RUN: llvm-nm -m %t/out | FileCheck --check-prefix=DYNAMICSYM %s

# Same thing should happen if _foo starts out as an Undefined.
# `-U _foo` being passed twice shouldn't have an effect either.
# RUN: %lld -lSystem %t/main.o -u _foo -U _foo -U _foo -u _bar -U _bar -U _baz -o %t/out
# RUN: llvm-objdump --macho --lazy-bind %t/out | FileCheck --check-prefix=DYNAMIC %s
# RUN: llvm-nm -m %t/out | FileCheck --check-prefix=DYNAMICSYM %s

# Unreferenced dynamic lookup symbols don't make it into the bind tables, but
# they do make it into the symbol table in ld64 if they're an undefined from -u
# for some reason. lld happens to have the same behavior when no explicit code
# handles this case, so match ld64's behavior.

# DYNAMIC-NOT: _bar
# DYNAMIC-NOT: _baz
# DYNAMIC: flat-namespace   _foo

# DYNAMICSYM:      (undefined) external _bar (dynamically looked up)
# DYNAMICSYM-NOT:      (undefined) external _bar (dynamically looked up)
# DYNAMICSYM-NEXT: (undefined) external _foo (dynamically looked up)

# Test with a Defined. Here, foo.o provides _foo and the symbol doesn't need
# to be imported.
# RUN: %lld -lSystem %t/main.o %t/foo.o -U _foo -o %t/out
# RUN: llvm-objdump --macho --lazy-bind %t/out | FileCheck --check-prefix=NOTDYNAMIC %s

# NOTDYNAMIC-NOT: _foo

# Here, foo.dylib provides _foo and the symbol doesn't need to be imported
# dynamically.
# RUN: %lld -lSystem %t/main.o %t/foo.dylib -U _foo -o %t/out
# RUN: llvm-objdump --macho --lazy-bind %t/out | FileCheck --check-prefix=TWOLEVEL %s
# RUN: llvm-nm -m %t/out | FileCheck --check-prefix=TWOLEVELSYM %s

# TWOLEVEL: foo              _foo
# TWOLEVELSYM: (undefined) external _foo (from foo)

# Test resolving dynamic lookup symbol with weak defined.
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos -o %t/weak-foo.o %t/weak-foo.s
# RUN: %lld -dylib -o %t/weak-foo.dylib %t/weak-foo.o -U _foo
# RUN: llvm-objdump --macho --bind --lazy-bind --weak-bind %t/weak-foo.dylib | FileCheck --check-prefix=WEAKDEF %s
# RUN: llvm-nm -m %t/weak-foo.dylib | FileCheck --check-prefix=WEAKDEFSYM %s
# WEAKDEF-NOT: _foo
# WEAKDEFSYM: weak external _foo

# Same if foo.dylib provides _foo weakly, except that the symbol is weak then.
# RUN: %lld -lSystem %t/main.o %t/weak-foo.dylib -U _foo -o %t/out
# RUN: llvm-objdump --macho --bind --lazy-bind --weak-bind %t/out | FileCheck --check-prefix=TWOLEVELWEAK %s
# RUN: llvm-nm -m %t/out | FileCheck --check-prefix=TWOLEVELWEAKSYM %s

# TWOLEVELWEAK-LABEL: Bind table:
# TWOLEVELWEAK:       __DATA        __la_symbol_ptr 0x[[#%x,ADDR:]]   pointer 0 weak-foo    _foo
# TWOLEVELWEAK-LABEL: Lazy bind table:
# TWOLEVELWEAK-NOT:   weak-foo         _foo
# TWOLEVELWEAK-LABEL: Weak bind table:
# TWOLEVELWEAK:       __DATA   __la_symbol_ptr      0x[[#ADDR]]       pointer 0 _foo

# TWOLEVELWEAKSYM: (undefined) weak external _foo (from weak-foo)

#--- foo.s
.globl _foo
_foo:
  ret

#--- weak-foo.s
.globl _foo
.weak_definition _foo
_foo:
  ret

#--- main.s
.globl _main
_main:
  callq _foo
  ret
