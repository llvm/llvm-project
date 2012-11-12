// RUN: %clang_cc1 %s -emit-llvm -o - 

typedef float __m128 __attribute__((__vector_size__(16)));
typedef long long __v2di __attribute__((__vector_size__(16)));
typedef int __v4si __attribute__((__vector_size__(16)));

__v2di  bar(void);
void foo(int X, __v4si *P) {
	*P = X == 2 ? bar() : bar();
}

