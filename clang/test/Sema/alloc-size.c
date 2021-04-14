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

int fail10 __attribute__((alloc_size(1))); //expected-warning{{'alloc_size' attribute only applies to non-K&R-style functions}}

void *fail11(void *a) __attribute__((alloc_size(1))); //expected-error{{'alloc_size' attribute argument may only refer to a function parameter of integer type}}

void *fail12(int a) __attribute__((alloc_size("abc"))); //expected-error{{'alloc_size' attribute requires parameter 1 to be an integer constant}}
void *fail12(int a) __attribute__((alloc_size(1, "abc"))); //expected-error{{'alloc_size' attribute requires parameter 2 to be an integer constant}}
void *fail13(int a) __attribute__((alloc_size(1U<<31))); //expected-error{{integer constant expression evaluates to value 2147483648 that cannot be represented in a 32-bit signed integer type}}

void *(*PR31453)(int)__attribute__((alloc_size(1)));

void *KR() __attribute__((alloc_size(1))); //expected-warning{{'alloc_size' attribute only applies to non-K&R-style functions}}

// Applying alloc_size to function pointers should work:
void *(__attribute__((alloc_size(1))) * func_ptr1)(int);
void *(__attribute__((alloc_size(1, 2))) func_ptr2)(int, int);

// TODO: according to GCC documentation the following should actually be the type
// “pointer to pointer to alloc_size attributed function returning void*” and should
// therefore be supported
void *(__attribute__((alloc_size(1))) **ptr_to_func_ptr)(int); // expected-warning{{'alloc_size' attribute only applies to non-K&R-style functions}}
// The following definitions apply the attribute to the pointer to the function pointer which should not be possible
void *(*__attribute__((alloc_size(1))) * ptr_to_func_ptr2)(int); // expected-warning{{'alloc_size' attribute only applies to non-K&R-style functions}}
void *(**__attribute__((alloc_size(1))) ptr_to_func_ptr2)(int);  // expected-warning{{'alloc_size' attribute only applies to non-K&R-style functions}}

// It should also work for typedefs:
typedef void *(__attribute__((alloc_size(1))) allocator_function_typdef)(int);
typedef void *(__attribute__((alloc_size(1, 2))) * allocator_function_typdef2)(int, int);
void *(__attribute__((alloc_size(1, 2))) * allocator_function_typdef3)(int, int);
// This typedef applies the alloc_size to the pointer to the function pointer and should not be allowed
void *(**__attribute__((alloc_size(1, 2))) * allocator_function_typdef4)(int, int); // expected-warning{{'alloc_size' attribute only applies to non-K&R-style functions}}

// We should not be warning when assigning function pointers with and without the alloc size attribute
// since it doesn't change the type of the function
typedef void *(__attribute__((alloc_size(1))) * my_malloc_fn_pointer_type)(int);
typedef void *(*my_other_malloc_fn_pointer_type)(int);
void *fn(int i);
__attribute__((alloc_size(1))) void *fn2(int i);

int main() {
  my_malloc_fn_pointer_type f = fn;
  my_other_malloc_fn_pointer_type f2 = fn;
  my_malloc_fn_pointer_type f3 = fn2;
  my_other_malloc_fn_pointer_type f4 = fn2;
}
