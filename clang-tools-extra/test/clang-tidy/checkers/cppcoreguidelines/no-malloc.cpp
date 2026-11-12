// RUN: %check_clang_tidy %s cppcoreguidelines-no-malloc %t

using size_t = __SIZE_TYPE__;

void *malloc(size_t size);
void *calloc(size_t num, size_t size);
void *realloc(void *ptr, size_t size);
void free(void *ptr);

void malloced_array() {
  int *array0 = (int *)malloc(sizeof(int) * 20);
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: do not manage memory manually; consider a container or a smart pointer [cppcoreguidelines-no-malloc]

  int *zeroed = (int *)calloc(20, sizeof(int));
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: do not manage memory manually; consider a container or a smart pointer [cppcoreguidelines-no-malloc]

  // reallocation memory, std::vector shall be used
  char *realloced = (char *)realloc(array0, 50 * sizeof(int));
  // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: do not manage memory manually; consider std::vector or std::string [cppcoreguidelines-no-malloc]

  // freeing memory the bad way
  free(realloced);
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
