# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %s -o %t.o
# RUN: ld.lld %t.o -o %t.so --shared --entry=func1.cold.1 --emit-relocs
# RUN: llvm-bolt -relocs %t.so -o %t -reorder-functions=hfsort+ \
# RUN:    -split-functions -reorder-blocks=ext-tsp -split-all-cold \
# RUN:    -dyno-stats -icf=1 -use-gnu-stack

# Check that an entry point is a cold symbol
# RUN: llvm-readelf -h %t.so > %t.log
# RUN: llvm-nm %t.so >> %t.log
# RUN: FileCheck %s --input-file %t.log
# CHECK: Entry point address: 0x[[#%X,ENTRY:]]
# CHECK: [[#%x,ENTRY]] {{.*}} func1.cold.1

.globl func1.cold.1
.type func1.cold.1,@function
func1.cold.1:
  .cfi_startproc
.L1:
		movq %rbx, %rdx
		jmp .L3
.L2:
		# exit(0)
		movq $60, %rax
		xorq %rdi, %rdi
		syscall
  .size func1.cold.1, .-func1.cold.1
  .cfi_endproc

.globl func1
.type func1,@function
func1:
  .cfi_startproc
.L3:
		movq %rax, %rdi
		jmp .L2
  call exit
  .size func1, .-func1
  .cfi_endproc
