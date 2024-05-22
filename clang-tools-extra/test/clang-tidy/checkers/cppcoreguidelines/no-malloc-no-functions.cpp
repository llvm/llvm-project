// RUN: %check_clang_tidy %s cppcoreguidelines-no-malloc %t \
// RUN: -config='{CheckOptions: \
// RUN:  {cppcoreguidelines-no-malloc.Allocations: "::malloc",\
// RUN:   cppcoreguidelines-no-malloc.Reallocations: "",\
// RUN:   cppcoreguidelines-no-malloc.Deallocations: ""}}' \
// RUN: --

// Just ensure, the check will not crash, when no functions shall be checked.

using size_t = __SIZE_TYPE__;

void *malloc(size_t size);

void malloced_array() {
  int *array0 = (int *)malloc(sizeof(int) * 20);
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: do not manage memory manually; consider a container or a smart pointer [cppcoreguidelines-no-malloc]
}
