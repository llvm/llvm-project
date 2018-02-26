// RUN: %clang_cc1 %s -verify

void *fail1(int a) __attribute__((alloc_size)); //expected-error{{'alloc_size' attribute takes at least 1 argument}}
void *fail2(int a) __attribute__((alloc_size())); //expected-error{{'alloc_size' attribute takes at least 1 argument}}

void *fail3(int a) __attribute__((alloc_size(0))); //expected-error{{'alloc_size' attribute parameter 1 is out of bounds}}
void *fail4(int a) __attribute__((alloc_size(2))); //expected-error{{'alloc_size' attribute parameter 1 is out of bounds}}

void *fail5(int a, int b) __attribute__((alloc_size(0, 1))); //expected-error{{'alloc_size' attribute parameter 1 is out of bounds}}
void *fail6(int a, int b) __attribute__((alloc_size(3, 1))); //expected-error{{'alloc_size' attribute parameter 1 is out of bounds}}

void *fail7(int a, int b) __attribute__((alloc_size(1, 0))); //expected-error{{'alloc_size' attribute parameter 2 is out of bounds}}
void *fail8(int a, int b) __attribute__((alloc_size(1, 3))); //expected-error{{'alloc_size' attribute parameter 2 is out of bounds}}

int fail9(int a) __attribute__((alloc_size(1))); //expected-warning{{'alloc_size' attribute only applies to return values that are pointers}}

int fail10 __attribute__((alloc_size(1))); //expected-warning{{'alloc_size' attribute only applies to functions}}

void *fail11(void *a) __attribute__((alloc_size(1))); //expected-error{{'alloc_size' attribute argument may only refer to a function parameter of integer type}}

void *fail12(int a) __attribute__((alloc_size("abc"))); //expected-error{{'alloc_size' attribute requires parameter 1 to be an integer constant}}
void *fail12(int a) __attribute__((alloc_size(1, "abc"))); //expected-error{{'alloc_size' attribute requires parameter 2 to be an integer constant}}
void *fail13(int a) __attribute__((alloc_size(1U<<31))); //expected-error{{integer constant expression evaluates to value 2147483648 that cannot be represented in a 32-bit signed integer type}}

int (*PR31453)(int) __attribute__((alloc_size(1))); //expected-warning{{'alloc_size' attribute only applies to functions}}
