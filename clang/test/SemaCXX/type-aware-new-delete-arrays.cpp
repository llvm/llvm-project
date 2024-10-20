// RUN: %clang_cc1 -fsyntax-only -verify %s -DNO_TADD -std=c++2c -fexperimental-cxx-type-aware-allocators -fexceptions -Wall -Wpedantic

namespace std {
  template <class T> struct type_identity {};
  enum class align_val_t : __SIZE_TYPE__ {};
  struct destroying_delete_t { explicit destroying_delete_t() = default; };
}

using size_t = __SIZE_TYPE__;

struct BasicTypeAwareArrayAllocator {
   template <typename T> void *operator new[](std::type_identity<T>, size_t) = delete;
   // expected-note@-1 {{candidate function [with T = BasicTypeAwareArrayAllocator] has been explicitly deleted}}
   template <typename T> void  operator delete[](std::type_identity<T>, void*) = delete;
   // expected-note@-1 {{'operator delete[]<BasicTypeAwareArrayAllocator>' has been explicitly marked deleted here}}
};
void *operator new[](std::type_identity<BasicTypeAwareArrayAllocator>, size_t);
void  operator delete[](std::type_identity<BasicTypeAwareArrayAllocator>, void*);

struct BasicTypeAwareNonArrayAllocator {
   template <typename T> void *operator new[](std::type_identity<T>, size_t);
   template <typename T> void  operator delete[](std::type_identity<T>, void*);
   void *operator new(size_t) = delete;
   void operator delete(void*) = delete;
};

struct WorkingTypeAwareAllocator {
   template <typename T> void *operator new[](std::type_identity<T>, size_t);
   template <typename T> void  operator delete[](std::type_identity<T>, void*);
};

void *operator new[](std::type_identity<WorkingTypeAwareAllocator>, size_t) = delete;
void  operator delete[](std::type_identity<WorkingTypeAwareAllocator>, void*) = delete;


void test() {
  BasicTypeAwareArrayAllocator *A0 = new BasicTypeAwareArrayAllocator[10]; // expected-error {{call to deleted function 'operator new[]'}}
  delete [] A0; // expected-error {{attempt to use a deleted function}}
  
  BasicTypeAwareNonArrayAllocator *A1 = new BasicTypeAwareNonArrayAllocator[10]; // ex
  delete [] A1;

  WorkingTypeAwareAllocator *A2 = new WorkingTypeAwareAllocator[10]; // expected-note {{allocated with 'new[]' here}}
  delete A2; // expected-warning {{'delete' applied to a pointer that was allocated with 'new[]'; did you mean 'delete[]'?}}

  WorkingTypeAwareAllocator *A3 = new WorkingTypeAwareAllocator; // expected-note {{allocated with 'new' here}}
  delete [] A3; // expected-warning {{'delete[]' applied to a pointer that was allocated with 'new'; did you mean 'delete'?}}

}
