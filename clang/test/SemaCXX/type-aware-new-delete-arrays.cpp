// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++26 -fexceptions    -fsized-deallocation    -faligned-allocation  -Wall -Wpedantic
// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++26 -fexceptions -fno-sized-deallocation    -faligned-allocation  -Wall -Wpedantic
// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++26 -fexceptions -fno-sized-deallocation -fno-aligned-allocation  -Wall -Wpedantic
// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++26 -fexceptions    -fsized-deallocation -fno-aligned-allocation  -Wall -Wpedantic

namespace std {
  template <class T> struct type_identity {};
  enum class align_val_t : __SIZE_TYPE__ {};
  struct destroying_delete_t { explicit destroying_delete_t() = default; };
}

using size_t = __SIZE_TYPE__;

struct BasicTypeAwareArrayAllocator {
   template <typename T> void *operator new[](std::type_identity<T>, size_t, std::align_val_t) = delete; // #1
   template <typename T> void  operator delete[](std::type_identity<T>, void*, size_t, std::align_val_t) = delete; // #2
};
void *operator new[](std::type_identity<BasicTypeAwareArrayAllocator>, size_t, std::align_val_t);
void  operator delete[](std::type_identity<BasicTypeAwareArrayAllocator>, void*, size_t, std::align_val_t);

struct BasicTypeAwareNonArrayAllocator {
   template <typename T> void *operator new[](std::type_identity<T>, size_t, std::align_val_t);
   template <typename T> void  operator delete[](std::type_identity<T>, void*, size_t, std::align_val_t);
   void *operator new(size_t) = delete;
   void operator delete(void*) = delete;
};

struct WorkingTypeAwareAllocator {
   template <typename T> void *operator new[](std::type_identity<T>, size_t, std::align_val_t);
   template <typename T> void  operator delete[](std::type_identity<T>, void*, size_t, std::align_val_t);
};

void *operator new[](std::type_identity<WorkingTypeAwareAllocator>, size_t, std::align_val_t) = delete;
void  operator delete[](std::type_identity<WorkingTypeAwareAllocator>, void*, size_t, std::align_val_t) = delete;


void test() {
  BasicTypeAwareArrayAllocator *A0 = new BasicTypeAwareArrayAllocator[10];
  // expected-error@-1 {{call to deleted function 'operator new[]'}}
  // expected-note@#1 {{candidate function [with T = BasicTypeAwareArrayAllocator] has been explicitly deleted}}
  delete [] A0;
  // expected-error@-1 {{attempt to use a deleted function}}
  // expected-note@#2 {{'operator delete[]<BasicTypeAwareArrayAllocator>' has been explicitly marked deleted here}}
  
  BasicTypeAwareNonArrayAllocator *A1 = new BasicTypeAwareNonArrayAllocator[10];
  delete [] A1;

  WorkingTypeAwareAllocator *A2 = new WorkingTypeAwareAllocator[10];
  // expected-note@-1 {{allocated with 'new[]' here}}
  delete A2;
  // expected-warning@-1 {{'delete' applied to a pointer that was allocated with 'new[]'; did you mean 'delete[]'?}}

  WorkingTypeAwareAllocator *A3 = new WorkingTypeAwareAllocator;
  // expected-note@-1 {{allocated with 'new' here}}
  delete [] A3;
  // expected-warning@-1 {{'delete[]' applied to a pointer that was allocated with 'new'; did you mean 'delete'?}}

}
