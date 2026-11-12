// RUN: %clang_cc1 -emit-llvm %s -o - -O | not grep alloca
// PR2094

int sad16_sse2(void *v, unsigned char *blk2, unsigned char *blk1,
               int stride, int h) {
    int ret;
    asm volatile( "%0 %1 %2 %3"
        : "+r" (h), "+r" (blk1), "+r" (blk2)
        : "r" ((long)stride));
    asm volatile("set %0 %1" : "=r"(ret) : "r"(blk1));
    return ret;
}
