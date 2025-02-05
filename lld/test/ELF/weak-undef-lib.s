# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t1.o
# RUN: echo -e '.globl foo\nfoo: ret' | \
# RUN:   llvm-mc -filetype=obj -triple=x86_64-pc-linux - -o %t2.o

# RUN: ld.lld -shared -o %t.so %t1.o --start-lib %t2.o
# RUN: llvm-readobj --dyn-syms %t.so | FileCheck %s

# RUN: ld.lld -pie -o %t %t1.o --start-lib %t2.o
# RUN: llvm-readobj --dyn-syms %t | FileCheck %s

# CHECK:      Name: foo
# CHECK-NEXT: Value: 0x0
# CHECK-NEXT: Size: 0
# CHECK-NEXT: Binding: Weak
# CHECK-NEXT: Type: None
# CHECK-NEXT: Other: 0
# CHECK-NEXT: Section: Undefined

# RUN: ld.lld -pie -o %t %t1.o --start-lib %t2.o --no-dynamic-linker
# RUN: llvm-readelf --dyn-syms %t | FileCheck %s --check-prefix=STATICPIE

# STATICPIE: Symbol table '.dynsym' contains 1

## -u specifies a STB_DEFAULT undefined symbol, so the definition from %t2.o is
## fetched.
# RUN: ld.lld -u foo %t1.o --start-lib %t2.o -o %t1
# RUN: llvm-readobj --syms %t1 | FileCheck %s --check-prefix=CHECK-U

# CHECK-U:      Name: foo
# CHECK-U:      Binding:
# CHECK-U-SAME:          Global
# CHECK-U:      Section:
# CHECK-U-SAME:          .text

.weak foo
call foo@PLT

.data
.quad foo
