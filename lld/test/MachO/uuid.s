# REQUIRES: x86
# UNSUPPORTED: system-windows
# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t/test.o
# RUN: %lld -lSystem %t/test.o -o %t/a
# RUN: %lld -lSystem %t/test.o -o %t/b
# RUN: llvm-dwarfdump --uuid %t/a | awk '{print $2}' > %t/uuida
# RUN: llvm-dwarfdump --uuid %t/b | awk '{print $2}' > %t/uuidb
# RUN: FileCheck %s < %t/uuida
# RUN: FileCheck %s < %t/uuidb
# RUN: not cmp %t/uuida %t/uuidb

## Ensure -final_output is used for universal binaries, which may be linked with
## temporary output file names
# RUN: %lld -lSystem %t/test.o -o %t/c -final_output %t/a
# RUN: llvm-dwarfdump --uuid %t/c | awk '{print $2}' > %t/uuidc
# RUN: cmp %t/uuida %t/uuidc

## Test disabling UUID generation
# RUN: %lld -lSystem %t/test.o -o %t/d -no_uuid
# RUN: llvm-dwarfdump --uuid %t/d | count 0

# CHECK: 4C4C44{{([[:xdigit:]]{2})}}-5555-{{([[:xdigit:]]{4})}}-A1{{([[:xdigit:]]{2})}}-{{([[:xdigit:]]{12})}}

.globl _main
_main:
  ret
