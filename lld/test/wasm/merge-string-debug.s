# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %s -o %t.o
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %p/Inputs/merge-string-debug2.s -o %t2.o

# RUN: wasm-ld %t.o %t2.o -o %t.wasm --no-entry
# RUN: llvm-readobj -x .debug_str %t.wasm | FileCheck %s --check-prefixes CHECK,CHECK-O1

# Check that we -r/--reclocatable can handle string merging too
# RUN: wasm-ld --relocatable %t.o %t2.o -o %t3.o
# RUN: wasm-ld -O1 %t3.o -o %t.wasm --no-entry
# RUN: llvm-readobj -x .debug_str %t.wasm | FileCheck %s --check-prefixes CHECK,CHECK-O1

# RUN: wasm-ld -O0 %t.o %t2.o -o %tO0.wasm --no-entry
# RUN: llvm-readobj -x .debug_str %tO0.wasm | FileCheck %s --check-prefixes CHECK,CHECK-O0
# RUN: llvm-readobj -x .debug_str_offsets %tO0.wasm | FileCheck %s --check-prefixes CHECK-OFFSETS

.section .debug_str,"S",@
.Linfo_string0:
  .asciz "clang version 13.0.0"
.Linfo_string1:
  .asciz "foobar"

.section .debug_other,"",@
  .int32 .Linfo_string0

.section .debug_str_offsets,"",@
  .int32 .Linfo_string0
  .int32 .Linfo_string0
  .int32 .Linfo_string0

# CHECK: Hex dump of section '.debug_str':

# CHECK-O0: 0x00000000 636c616e 67207665 7273696f 6e203133 clang version 13
# CHECK-O0: 0x00000010 2e302e30 00666f6f 62617200 636c616e .0.0.foobar.clan
# CHECK-O0: 0x00000020 67207665 7273696f 6e203133 2e302e30 g version 13.0.0
# CHECK-O0: 0x00000030 00626172 00666f6f 00                .bar.foo.

# CHECK-O1: 0x00000000 666f6f62 61720066 6f6f0063 6c616e67 foobar.foo.clang
# CHECK-O1: 0x00000010 20766572 73696f6e 2031332e 302e3000  version 13.0.0.

# CHECK-OFFSETS: Hex dump of section '.debug_str_offsets':
# CHECK-OFFSETS: 0x00000000 00000000 00000000 00000000          ............
