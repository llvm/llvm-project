// RUN: %check_clang_tidy %s cppcoreguidelines-no-malloc %t \
// RUN: -config='{CheckOptions: \
// RUN:  [{key: cppcoreguidelines-no-malloc.Allocations, value: "::malloc;::align_malloc;::calloc"},\
// RUN:   {key: cppcoreguidelines-no-malloc.Reallocations, value: "::realloc;::align_realloc"},\
// RUN:   {key: cppcoreguidelines-no-malloc.Deallocations, value: "::free;::align_free"}]}' \
// RUN: --

using size_t = __SIZE_TYPE__;

void *malloc(size_t size);
void *align_malloc(size_t size, unsigned short alignment);
void *calloc(size_t num, size_t size);
void *realloc(void *ptr, size_t size);
void *align_realloc(void *ptr, size_t size, unsigned short alignment);
void free(void *ptr);
void *align_free(void *ptr);

void malloced_array() {
  int *array0 = (int *)malloc(sizeof(int) * 20);
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: do not manage memory manually; consider a container or a smart pointer [cppcoreguidelines-no-malloc]

  int *zeroed = (int *)calloc(20, sizeof(int));
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: do not manage memory manually; consider a container or a smart pointer [cppcoreguidelines-no-malloc]

  int *aligned = (int *)align_malloc(20 * sizeof(int), 16);
  // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: do not manage memory manually; consider a container or a smart pointer [cppcoreguidelines-no-malloc]

  // reallocation memory, std::vector shall be used
  char *realloced = (char *)realloc(array0, 50 * sizeof(int));
  // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: do not manage memory manually; consider std::vector or std::string [cppcoreguidelines-no-malloc]

  char *align_realloced = (char *)align_realloc(aligned, 50 * sizeof(int), 16);
  // CHECK-MESSAGES: :[[@LINE-1]]:35: warning: do not manage memory manually; consider std::vector or std::string [cppcoreguidelines-no-malloc]

  // freeing memory the bad way
  free(realloced);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not manage memory manually; use RAII [cppcoreguidelines-no-malloc]

  align_free(align_realloced);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not manage memory manually; use RAII [cppcoreguidelines-no-malloc]
  
  // check if a call to malloc as function argument is found as well
  free(malloc(20));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not manage memory manually; use RAII [cppcoreguidelines-no-malloc]
  // CHECK-MESSAGES: :[[@LINE-2]]:8: warning: do not manage memory manually; consider a container or a smart pointer [cppcoreguidelines-no-malloc]
}

/// newing an array is still not good, but not relevant to this checker
void newed_array() {
  int *new_array = new int[10]; // OK(1)
}

void arbitrary_call() {
  // we dont want every function to raise the warning even if malloc is in the name
  malloced_array(); // OK(2)

  // completely unrelated function call to malloc
  newed_array(); // OK(3)
}
