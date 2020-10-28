# REQUIRES: x86, shell
# RUN: mkdir -p %t
# RUN: echo ".globl _foo; _foo: ret" | llvm-mc -filetype=obj -triple=x86_64-apple-darwin -o %t/foo.o
# RUN: mkdir -p %t/Foo.framework/Versions/A
# RUN: %lld -dylib -install_name %t/Foo.framework/Versions/A/Foo %t/foo.o -o %t/Foo.framework/Versions/A/Foo
# RUN: %lld -dylib -install_name %t/Foo.framework/Versions/A/Foobar %t/foo.o -o %t/Foo.framework/Versions/A/Foobar
# RUN: ln -sf %t/Foo.framework/Versions/A %t/Foo.framework/Versions/Current
# RUN: ln -sf %t/Foo.framework/Versions/Current/Foo %t/Foo.framework/Foo

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin -o %t/test.o %s
# RUN: %lld -lSystem -F%t -framework Foo %t/test.o -o %t/test
# RUN: llvm-objdump --macho --lazy-bind %t/test | FileCheck %s --check-prefix=NOSUFFIX
# NOSUFFIX: __DATA __la_symbol_ptr 0x{{[0-9a-f]*}} {{.*}}Foo _foo

# RUN: %lld -lSystem -F%t -framework Foo,baz %t/test.o -o %t/test-wrong-suffix
# RUN: llvm-objdump --macho --lazy-bind %t/test-wrong-suffix | FileCheck %s --check-prefix=NOSUFFIX

# RUN: %lld -lSystem -F%t -framework Foo,bar %t/test.o -o %t/test-suffix
# RUN: llvm-objdump --macho --lazy-bind %t/test-suffix | FileCheck %s --check-prefix=SUFFIX
# SUFFIX: __DATA __la_symbol_ptr 0x{{[0-9a-f]*}} {{.*}}Foobar _foo

.globl _main
.text
_main:
  sub $8, %rsp # 16-byte-align the stack; dyld checks for this
  callq _foo
  mov $0, %rax
  add $8, %rsp
  ret
