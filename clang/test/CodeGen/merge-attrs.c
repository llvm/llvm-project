// RUN: %clang_cc1 %s -emit-llvm -o %t

void *malloc(__SIZE_TYPE__ size) __attribute__ ((__nothrow__));

inline static void __zend_malloc(int) {
    malloc(1);
}

void *malloc(__SIZE_TYPE__ size) __attribute__ ((__nothrow__));

void fontFetch(void) {
    __zend_malloc(1);
}
