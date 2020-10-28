# REQUIRES: x86
# RUN: mkdir -p %t

## This tests examines the effect of .weak_definition on symbols in an archive
## that are not referenced directly, but which are still loaded due to some
## other symbol in the archive member being referenced.
##
## In this particular test, _foo isn't referenced directly, but both archives
## will be fetched when linking against the main test file due to its references
## to _bar and _baz.

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t/test.o
# RUN: echo ".globl _foo, _bar; .section __TEXT,nonweak; _bar: _foo:" | llvm-mc -filetype=obj -triple=x86_64-apple-darwin -o %t/foo.o
# RUN: echo ".globl _foo, _baz; .weak_definition _foo; .section __TEXT,weak; _baz: _foo:" | llvm-mc -filetype=obj -triple=x86_64-apple-darwin -o %t/weakfoo.o

# RUN: rm -f %t/foo.a
# RUN: llvm-ar --format=darwin rcs %t/foo.a %t/foo.o
# RUN: rm -f %t/weakfoo.a
# RUN: llvm-ar --format=darwin rcs %t/weakfoo.a %t/weakfoo.o

# PREFER-NONWEAK-OBJECT: O __TEXT,nonweak _foo

# RUN: %lld -lSystem -o %t/weak-nonweak-archives -L%t %t/weakfoo.a %t/foo.a %t/test.o
# RUN: llvm-objdump --syms %t/weak-nonweak-archives | FileCheck %s --check-prefix=PREFER-NONWEAK-OBJECT
# RUN: %lld -lSystem -o %t/nonweak-weak-archives -L%t %t/foo.a %t/weakfoo.a %t/test.o
# RUN: llvm-objdump --syms %t/nonweak-weak-archives | FileCheck %s --check-prefix=PREFER-NONWEAK-OBJECT

# RUN: %lld -lSystem -o %t/weak-nonweak-objs -L%t %t/weakfoo.o %t/foo.o %t/test.o
# RUN: llvm-objdump --syms %t/weak-nonweak-objs | FileCheck %s --check-prefix=PREFER-NONWEAK-OBJECT
# RUN: %lld -lSystem -o %t/nonweak-weak-objs -L%t %t/foo.o %t/weakfoo.o %t/test.o
# RUN: llvm-objdump --syms %t/nonweak-weak-objs | FileCheck %s --check-prefix=PREFER-NONWEAK-OBJECT

# RUN: %lld -lSystem -o %t/weak-obj-nonweak-ar -L%t %t/weakfoo.o %t/foo.a %t/test.o
# RUN: llvm-objdump --syms %t/weak-obj-nonweak-ar | FileCheck %s --check-prefix=PREFER-NONWEAK-OBJECT
# RUN: %lld -lSystem -o %t/nonweak-ar-weak-obj -L%t %t/foo.a %t/weakfoo.o %t/test.o
# RUN: llvm-objdump --syms %t/nonweak-ar-weak-obj | FileCheck %s --check-prefix=PREFER-NONWEAK-OBJECT

.globl _main
_main:
  callq _bar
  callq _baz
  ret
