// We would need to read special register ctr_el0 to find out L1 dcache size.
// This value is a good estimate based on a real aarch64 machine.
#define LLVM_LIBC_CACHELINE_SIZE 64
