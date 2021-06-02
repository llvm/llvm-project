# REQUIRES: x86
# RUN: rm -rf %t; mkdir -p %t

## Create a libsuper that has libgoodbye as a sub-library, which in turn has
## libhello as another sub-library.
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %p/Inputs/libhello.s \
# RUN:   -o %t/libhello.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %p/Inputs/libgoodbye.s \
# RUN:   -o %t/libgoodbye.o
# RUN: echo "" | llvm-mc -filetype=obj -triple=x86_64-apple-darwin -o %t/libsuper.o
# RUN: %lld -dylib %t/libhello.o -o %t/libhello.dylib
# RUN: %lld -dylib -L%t -sub_library libhello -lhello \
# RUN:   %t/libgoodbye.o -o %t/libgoodbye.dylib
# RUN: %lld -dylib -L%t -sub_library libgoodbye -lgoodbye -install_name \
# RUN:   @executable_path/libsuper.dylib %t/libsuper.o -o %t/libsuper.dylib


## Check that they have the appropriate LC_REEXPORT_DYLIB commands, and that
## NO_REEXPORTED_DYLIBS is (un)set as appropriate.

# RUN: llvm-otool -hv %t/libhello.dylib | \
# RUN:     FileCheck --check-prefix=HELLO-HEADERS %s
# HELLO-HEADERS: NO_REEXPORTED_DYLIBS

# RUN: llvm-otool -l %t/libgoodbye.dylib | FileCheck %s \
# RUN:   --check-prefix=REEXPORT-HEADERS -DPATH=%t/libhello.dylib

# RUN: llvm-otool -l %t/libsuper.dylib | FileCheck %s \
# RUN:   --check-prefix=REEXPORT-HEADERS -DPATH=%t/libgoodbye.dylib

# RUN: %lld -dylib -L%t -reexport-lgoodbye -install_name \
# RUN:   @executable_path/libsuper.dylib %t/libsuper.o -o %t/libsuper.dylib
# RUN: llvm-otool -l %t/libsuper.dylib | FileCheck %s \
# RUN:   --check-prefix=REEXPORT-HEADERS -DPATH=%t/libgoodbye.dylib
# RUN: %lld -dylib -reexport_library %t/libgoodbye.dylib -install_name \
# RUN:   @executable_path/libsuper.dylib %t/libsuper.o -o %t/libsuper.dylib
# RUN: llvm-otool -l %t/libsuper.dylib | FileCheck %s \
# RUN:   --check-prefix=REEXPORT-HEADERS -DPATH=%t/libgoodbye.dylib

# REEXPORT-HEADERS-NOT: NO_REEXPORTED_DYLIBS
# REEXPORT-HEADERS:     cmd     LC_REEXPORT_DYLIB
# REEXPORT-HEADERS-NOT: Load command
# REEXPORT-HEADERS:     name    [[PATH]]

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t/sub-library.o
# RUN: %lld -o %t/sub-library -L%t -lsuper %t/sub-library.o

# RUN: llvm-objdump --macho --bind %t/sub-library | FileCheck %s
# CHECK-LABEL: Bind table:
# CHECK-DAG:   __DATA_CONST __got {{.*}} libsuper _hello_world
# CHECK-DAG:   __DATA_CONST __got {{.*}} libsuper _goodbye_world


## Check that we fail gracefully if the sub-library is missing
# RUN: not %lld -dylib -o %t/sub-library -sub_library libmissing %t/sub-library.o 2>&1 \
# RUN:   | FileCheck %s --check-prefix=MISSING-SUB-LIBRARY
# MISSING-SUB-LIBRARY: error: -sub_library libmissing does not match a supplied dylib
# RUN: rm -f %t/libgoodbye.dylib
# RUN: not %lld -o %t/sub-library -L%t -lsuper %t/sub-library.o 2>&1 \
# RUN:  | FileCheck %s --check-prefix=MISSING-REEXPORT -DDIR=%t
# MISSING-REEXPORT: error: unable to locate re-export with install name [[DIR]]/libgoodbye.dylib


## We can match dylibs without extensions too.
# RUN: mkdir -p %t/Hello.framework
# RUN: %lld -dylib %t/libhello.o -o %t/Hello.framework/Hello
# RUN: %lld -dylib -o %t/libgoodbye2.dylib -sub_library Hello %t/Hello.framework/Hello %t/libgoodbye.o
# RUN: llvm-otool -l %t/libgoodbye2.dylib | FileCheck %s \
# RUN:   --check-prefix=REEXPORT-HEADERS -DPATH=%t/Hello.framework/Hello

## -sub_umbrella works almost identically...
# RUN: %lld -dylib -o %t/libgoodbye3.dylib -sub_umbrella Hello %t/Hello.framework/Hello %t/libgoodbye.o
# RUN: llvm-otool -l %t/libgoodbye3.dylib | FileCheck %s \
# RUN:   --check-prefix=REEXPORT-HEADERS -DPATH=%t/Hello.framework/Hello

# RUN: %lld -dylib -o %t/libgoodbye3.dylib -F %t -framework Hello -sub_umbrella Hello %t/libgoodbye.o
# RUN: llvm-otool -l %t/libgoodbye3.dylib | FileCheck %s \
# RUN:   --check-prefix=REEXPORT-HEADERS -DPATH=%t/Hello.framework/Hello

# RUN: %lld -dylib -o %t/libgoodbye3.dylib -F %t -reexport_framework Hello %t/libgoodbye.o
# RUN: llvm-otool -l %t/libgoodbye3.dylib | FileCheck %s \
# RUN:   --check-prefix=REEXPORT-HEADERS -DPATH=%t/Hello.framework/Hello

## But it doesn't match .dylib extensions:
# RUN: not %lld -dylib -L%t -sub_umbrella libhello -lhello %t/libgoodbye.o \
# RUN:   -o %t/libgoodbye.dylib 2>&1 | FileCheck %s --check-prefix=MISSING-FRAMEWORK
# MISSING-FRAMEWORK: error: -sub_umbrella libhello does not match a supplied dylib

.text
.globl _main

_main:
  movl $0x2000004, %eax # write() syscall
  mov $1, %rdi # stdout
  movq _hello_world@GOTPCREL(%rip), %rsi
  mov $13, %rdx # length of str
  syscall
  mov $0, %rax

  movl $0x2000004, %eax # write() syscall
  mov $1, %rdi # stdout
  movq _goodbye_world@GOTPCREL(%rip), %rsi
  mov $15, %rdx # length of str
  syscall
  mov $0, %rax
  ret
