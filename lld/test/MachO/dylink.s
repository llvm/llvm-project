# REQUIRES: x86
# RUN: mkdir -p %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %p/Inputs/libhello.s \
# RUN:   -o %t/libhello.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %p/Inputs/libgoodbye.s \
# RUN:   -o %t/libgoodbye.o
# RUN: %lld -dylib -install_name @executable_path/libhello.dylib \
# RUN:   -compatibility_version 10 -current_version 11 \
# RUN:   %t/libhello.o -o %t/libhello.dylib
# RUN: %lld -dylib -install_name \
# RUN:   @executable_path/libgoodbye.dylib %t/libgoodbye.o -o %t/libgoodbye.dylib

## Make sure we are using the export trie and not the symbol table when linking
## against these dylibs.
# RUN: llvm-strip %t/libhello.dylib
# RUN: llvm-strip %t/libgoodbye.dylib
# RUN: llvm-nm %t/libhello.dylib 2>&1 | FileCheck %s --check-prefix=NOSYM
# RUN: llvm-nm %t/libgoodbye.dylib 2>&1 | FileCheck %s --check-prefix=NOSYM
# NOSYM: no symbols

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t/dylink.o
# RUN: %lld -o %t/dylink -L%t -lhello -lgoodbye %t/dylink.o
# RUN: llvm-objdump --bind -d --no-show-raw-insn %t/dylink | FileCheck %s

# CHECK: movq [[#%u, HELLO_OFF:]](%rip), %rsi
# CHECK-NEXT: [[#%x, HELLO_RIP:]]:

# CHECK: movq [[#%u, HELLO_ITS_ME_OFF:]](%rip), %rsi
# CHECK-NEXT: [[#%x, HELLO_ITS_ME_RIP:]]:

# CHECK: pushq [[#%u, GOODBYE_OFF:]](%rip)
# CHECK-NEXT: [[#%x, GOODBYE_RIP:]]: popq %rsi

# CHECK-LABEL: Bind table:
# CHECK-DAG: __DATA_CONST __got  0x{{0*}}[[#%x, HELLO_RIP + HELLO_OFF]]               pointer 0   libhello   _hello_world
# CHECK-DAG: __DATA_CONST __got  0x{{0*}}[[#%x, HELLO_ITS_ME_RIP + HELLO_ITS_ME_OFF]] pointer 0   libhello   _hello_its_me
# CHECK-DAG: __DATA_CONST __got  0x{{0*}}[[#%x, GOODBYE_RIP + GOODBYE_OFF]]           pointer 0   libgoodbye _goodbye_world
# CHECK-DAG: __DATA       __data 0x[[#%x, DATA_ADDR:]]                                pointer 0   libhello   _hello_world
# CHECK-DAG: __DATA       __data 0x{{0*}}[[#%x, DATA_ADDR + 8]]                       pointer 8   libhello   _hello_its_me
# CHECK-DAG: __DATA       __data 0x{{0*}}[[#%x, DATA_ADDR + 16]]                      pointer -15 libgoodbye _goodbye_world

# RUN: llvm-nm -m %t/dylink | FileCheck --check-prefix=NM %s

# NM-DAG: _goodbye_world (from libgoodbye)
# NM-DAG: _hello_its_me (from libhello)
# NM-DAG: _hello_world (from libhello)

# RUN: llvm-objdump --macho --all-headers %t/dylink | FileCheck %s \
# RUN:   --check-prefix=LOAD --implicit-check-not LC_LOAD_DYLIB
## Check that we don't create duplicate LC_LOAD_DYLIBs.
# RUN: %lld -o %t/dylink -L%t -lhello -lhello -lgoodbye -lgoodbye %t/dylink.o
# RUN: llvm-objdump --macho --all-headers %t/dylink | FileCheck %s \
# RUN:   --check-prefix=LOAD --implicit-check-not LC_LOAD_DYLIB

# LOAD:                        cmd LC_LOAD_DYLIB
# LOAD-NEXT:               cmdsize
# LOAD-NEXT:                  name @executable_path/libhello.dylib
# LOAD-NEXT:            time stamp
# LOAD-NEXT:       current version 11.0.0
# LOAD-NEXT: compatibility version 10.0.0
# LOAD:                        cmd LC_LOAD_DYLIB
# LOAD-NEXT:               cmdsize
# LOAD-NEXT:                  name @executable_path/libgoodbye.dylib

.section __TEXT,__text
.globl _main

_main:
  movl $0x2000004, %eax # write() syscall
  mov $1, %rdi # stdout
  movq _hello_world@GOTPCREL(%rip), %rsi
  mov $13, %rdx # length of str
  syscall

  movl $0x2000004, %eax # write() syscall
  mov $1, %rdi # stdout
  movq _hello_its_me@GOTPCREL(%rip), %rsi
  mov $15, %rdx # length of str
  syscall

  movl $0x2000004, %eax # write() syscall
  mov $1, %rdi # stdout
  pushq _goodbye_world@GOTPCREL(%rip)
  popq %rsi
  mov $15, %rdx # length of str
  syscall
  mov $0, %rax
  ret

.data
.quad _hello_world
.quad _hello_its_me + 0x8
.quad _goodbye_world - 0xf
