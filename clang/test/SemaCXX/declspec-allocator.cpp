// RUN: %clang_cc1 -fms-compatibility -triple x86_64-windows-msvc -std=c++14 -fms-extensions -fms-compatibility-version=19.00 -verify %s

__declspec(allocator) int err_on_data; // expected-warning {{'allocator' attribute only applies to functions}}
__declspec(allocator) struct ErrOnStruct1; // expected-warning {{place it after "struct" to apply attribute}}
struct __declspec(allocator) ErrOnStruct2 {}; // expected-warning {{'allocator' attribute only applies to functions}}
__declspec(allocator) void err_on_ret_void(); // expected-warning {{not a pointer or reference type}}
__declspec(allocator) int err_on_ret_int(); // expected-warning {{not a pointer or reference type}}
__declspec(allocator) void *accept_on_ptr1();
__declspec(allocator) void *accept_on_ptr2(size_t);
void * __declspec(allocator) accept_on_ptr3(size_t); // expected-error {{expected unqualified-id}}

struct Foo { int x; };
__declspec(allocator) Foo *accept_nonvoid_ptr(size_t);
