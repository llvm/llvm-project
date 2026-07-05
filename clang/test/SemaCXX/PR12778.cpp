// RUN: %clang_cc1 -fsyntax-only -Wno-dynamic-exception-spec -verify %s

void operator delete() throw(void*); // expected-error{{'operator delete' must have at least one parameter}}
void* allocate(int __n) {
   return ::operator new(__n);
}

