#pragma clang system_header

static inline void memcpy(void *__restrict__ dst, void *__restrict__ src, int size) {
    __builtin_memcpy(dst, src, size);
}

static inline void* memcpy2(void *__restrict__ dst, void *__restrict__ src, int size) {
    return __builtin_memcpy(dst, src, size);
}

static inline void* memcpy3(void *__restrict__ dst, void *__restrict__ src, int size) {
    void * tmp = __builtin_memcpy(dst, src, size);
    return tmp;
}

static inline void* malloc(int size) {
    return __builtin_malloc(size);
}

static inline void* malloc2(int size) {
    void *tmp = __builtin_malloc(size);
    return tmp;
}

