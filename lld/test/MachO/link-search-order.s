# REQUIRES: x86

# RUN: mkdir -p %t
#
# RUN: llvm-mc -filetype obj -triple x86_64-apple-darwin %p/Inputs/libhello.s -o %t/hello.o
# RUN: lld -flavor darwinnew -arch x86_64 -dylib -install_name @executable_path/libhello.dylib %t/hello.o -o %t/libhello.dylib
#
# RUN: llvm-mc -filetype obj -triple x86_64-apple-darwin %p/Inputs/libgoodbye.s -o %t/goodbye.o
# RUN: lld -flavor darwinnew -arch x86_64 -dylib -install_name @executable_path/libgoodbye.dylib %t/goodbye.o -o %t/libgoodbye.dylib
# RUN: llvm-ar --format=darwin crs %t/libgoodbye.a %t/goodbye.o
#
# RUN: llvm-mc -filetype obj -triple x86_64-apple-darwin %s -o %t/test.o
# RUN: lld -flavor darwinnew -arch x86_64 -L%S/Inputs/MacOSX.sdk/usr/lib -o %t/test -Z -L%t -lhello -lgoodbye -lSystem %t/test.o
#
# RUN: llvm-objdump --macho --dylibs-used %t/test | FileCheck %s

# CHECK: @executable_path/libhello.dylib
# CHECK: @executable_path/libgoodbye.dylib
# CHECK: /usr/lib/libSystem.B.dylib

.section __TEXT,__text
.global _main

_main:
  movl $0x2000004, %eax                         # write()
  mov $1, %rdi                                  # stdout
  movq _hello_world@GOTPCREL(%rip), %rsi
  mov $13, %rdx                                 # length
  syscall

  movl $0x2000004, %eax                         # write()
  mov $1, %rdi                                  # stdout
  movq _hello_its_me@GOTPCREL(%rip), %rsi
  mov $15, %rdx                                 # length
  syscall

  movl $0x2000004, %eax                         # write()
  mov $1, %rdi                                  # stdout
  movq _goodbye_world@GOTPCREL(%rip), %rsi
  mov $15, %rdx                                 # length
  syscall
  mov $0, %rax
  ret
