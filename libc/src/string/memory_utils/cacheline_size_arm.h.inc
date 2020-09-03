// Cache line sizes for ARM: These values are not strictly correct since
// cache line sizes depend on implementations, not architectures.  There
// are even implementations with cache line sizes configurable at boot
// time.
#if defined(__ARM_ARCH_5T__) 
#define LLVM_LIBC_CACHELINE_SIZE 32
#elif defined(__ARM_ARCH_7A__)
#define LLVM_LIBC_CACHELINE_SIZE 64
#endif
