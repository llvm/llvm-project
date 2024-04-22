
// TODO: license block
#ifndef LLVM_LIBC_SRC_SETJMP_X86_64_SETJMP_H
#define LLVM_LIBC_SRC_SETJMP_X86_64_SETJMP_H

#ifdef __ASSEMBLER__
#define UL(x) x
#else
#define UL(x) x##UL
#endif

// Brittle! Changing the layout of __jmp_buf will break this!
#define RBX_OFFSET UL(0)
#define RBP_OFFSET UL(8)
#define R12_OFFSET UL(16)
#define R13_OFFSET UL(24)
#define R14_OFFSET UL(32)
#define R15_OFFSET UL(40)
#define RSP_OFFSET UL(48)
#define RIP_OFFSET UL(56)

#endif // LLVM_LIBC_SRC_SETJMP_X86_64_SETJMP_H
