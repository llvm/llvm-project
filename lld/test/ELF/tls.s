// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t
// RUN: ld.lld %t -o %tout
// RUN: llvm-readobj -S -l --symbols %tout | FileCheck %s
// RUN: llvm-objdump --no-print-imm-hex -d %tout | FileCheck %s --check-prefix=DIS

/// Reject local-exec TLS relocations for -shared, regardless of the preemptibility.
// RUN: not ld.lld -shared %t -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR
// RUN: not ld.lld -shared -Bsymbolic %t -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR

// ERR:       error: relocation R_X86_64_TPOFF32 against a cannot be used with -shared
// ERR-NEXT:  defined in {{.*}}
// ERR-NEXT:  referenced by {{.*}}:(.text+0x4)
// ERR-EMPTY:
// ERR-NEXT:  error: relocation R_X86_64_TPOFF32 against b cannot be used with -shared
// ERR-NEXT:  defined in {{.*}}
// ERR-NEXT:  referenced by {{.*}}:(.text+0xc)
// ERR-EMPTY:
// ERR-NEXT:  error: relocation R_X86_64_TPOFF32 against c cannot be used with -shared
// ERR-NEXT:  defined in {{.*}}
// ERR-NEXT:  referenced by {{.*}}:(.text+0x14)
// ERR-EMPTY:
// ERR-NEXT:  error: relocation R_X86_64_TPOFF32 against d cannot be used with -shared
// ERR-NEXT:  defined in {{.*}}
// ERR-NEXT:  referenced by {{.*}}:(.text+0x1c)

.global _start
_start:
  movl %fs:a@tpoff, %eax
  movl %fs:b@tpoff, %eax
  movl %fs:c@tpoff, %eax
  movl %fs:d@tpoff, %eax

  .global a
	.section	.tbss,"awT",@nobits
a:
	.long	0

  .global b
	.section	.tdata,"awT",@progbits
b:
	.long	1

  .global c
	.section	.thread_bss,"awT",@nobits
c:
	.long	0

  .global d
	.section	.thread_data,"awT",@progbits
d:
	.long	2

// CHECK:          Name: .tdata
// CHECK-NEXT:     Type: SHT_PROGBITS
// CHECK-NEXT:     Flags [
// CHECK-NEXT:       SHF_ALLOC
// CHECK-NEXT:       SHF_TLS
// CHECK-NEXT:       SHF_WRITE
// CHECK-NEXT:     ]
// CHECK-NEXT:     Address: [[TDATA_ADDR:0x.*]]
// CHECK-NEXT:     Offset:
// CHECK-NEXT:     Size: 4
// CHECK-NEXT:     Link:
// CHECK-NEXT:     Info:
// CHECK-NEXT:     AddressAlignment:
// CHECK-NEXT:     EntrySize:
// CHECK-NEXT:   }
// CHECK-NEXT:   Section {
// CHECK-NEXT:     Index:
// CHECK-NEXT:     Name: .thread_data
// CHECK-NEXT:     Type: SHT_PROGBITS
// CHECK-NEXT:     Flags [
// CHECK-NEXT:       SHF_ALLOC
// CHECK-NEXT:       SHF_TLS
// CHECK-NEXT:       SHF_WRITE
// CHECK-NEXT:     ]
// CHECK-NEXT:     Address:
// CHECK-NEXT:     Offset:
// CHECK-NEXT:     Size: 4
// CHECK-NEXT:     Link:
// CHECK-NEXT:     Info:
// CHECK-NEXT:     AddressAlignment:
// CHECK-NEXT:     EntrySize:
// CHECK-NEXT:   }
// CHECK-NEXT:   Section {
// CHECK-NEXT:     Index:
// CHECK-NEXT:     Name: .tbss
// CHECK-NEXT:     Type: SHT_NOBITS
// CHECK-NEXT:     Flags [
// CHECK-NEXT:       SHF_ALLOC
// CHECK-NEXT:       SHF_TLS
// CHECK-NEXT:       SHF_WRITE
// CHECK-NEXT:     ]
// CHECK-NEXT:     Address: [[TBSS_ADDR:0x.*]]
// CHECK-NEXT:     Offset:
// CHECK-NEXT:     Size: 4
// CHECK-NEXT:     Link:
// CHECK-NEXT:     Info:
// CHECK-NEXT:     AddressAlignment:
// CHECK-NEXT:     EntrySize:
// CHECK-NEXT:   }
// CHECK-NEXT:   Section {
// CHECK-NEXT:     Index:
// CHECK-NEXT:     Name: .thread_bss
// CHECK-NEXT:     Type: SHT_NOBITS
// CHECK-NEXT:     Flags [
// CHECK-NEXT:       SHF_ALLOC
// CHECK-NEXT:       SHF_TLS
// CHECK-NEXT:       SHF_WRITE
// CHECK-NEXT:     ]

// 0x2021F4 = TBSS_ADDR + 4

// CHECK-NEXT:     Address: 0x2021F4
// CHECK-NEXT:     Offset:
// CHECK-NEXT:     Size: 4
// CHECK-NEXT:     Link:
// CHECK-NEXT:     Info:
// CHECK-NEXT:     AddressAlignment:
// CHECK-NEXT:     EntrySize:
// CHECK-NEXT:   }

// CHECK:      ProgramHeaders [
// CHECK:          Type: PT_LOAD
// CHECK:          Type: PT_LOAD
// CHECK:          Type: PT_LOAD
// CHECK:          FileSize: 8
// CHECK-NEXT:     MemSize: [[#]]
// CHECK-NEXT:     Flags [
// CHECK-NEXT:       PF_R
// CHECK-NEXT:       PF_W
// CHECK-NEXT:     ]
// CHECK:          Type: PT_TLS
// CHECK-NEXT:     Offset:
// CHECK-NEXT:     VirtualAddress: [[TDATA_ADDR]]
// CHECK-NEXT:     PhysicalAddress: [[TDATA_ADDR]]
// CHECK-NEXT:     FileSize: 8
// CHECK-NEXT:     MemSize: 16
// CHECK-NEXT:     Flags [
// CHECK-NEXT:       PF_R
// CHECK-NEXT:     ]
// CHECK-NEXT:     Alignment:
// CHECK-NEXT:   }

// CHECK:      Symbols [
// CHECK:          Name: a
// CHECK-NEXT:     Value: 0x8
// CHECK-NEXT:     Size:
// CHECK-NEXT:     Binding: Global
// CHECK-NEXT:     Type: TLS
// CHECK-NEXT:     Other: 0
// CHECK-NEXT:     Section: .tbss
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: b
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:     Size:
// CHECK-NEXT:     Binding: Global
// CHECK-NEXT:     Type: TLS
// CHECK-NEXT:     Other: 0
// CHECK-NEXT:     Section: .tdata
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: c
// CHECK-NEXT:     Value: 0xC
// CHECK-NEXT:     Size:
// CHECK-NEXT:     Binding: Global
// CHECK-NEXT:     Type: TLS
// CHECK-NEXT:     Other: 0
// CHECK-NEXT:     Section: .thread_bss
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: d
// CHECK-NEXT:     Value: 0x4
// CHECK-NEXT:     Size:
// CHECK-NEXT:     Binding: Global
// CHECK-NEXT:     Type: TLS
// CHECK-NEXT:     Other: 0
// CHECK-NEXT:     Section: .thread_data
// CHECK-NEXT:   }

// DIS:      Disassembly of section .text:
// DIS-EMPTY:
// DIS-NEXT: <_start>:
// DIS-NEXT:   movl    %fs:-8, %eax
// DIS-NEXT:   movl    %fs:-16, %eax
// DIS-NEXT:   movl    %fs:-4, %eax
// DIS-NEXT:   movl    %fs:-12, %eax
