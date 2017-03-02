// RUN: %check_clang_tidy %s cppcoreguidelines-no-malloc %t \
// RUN: -config='{CheckOptions: \
// RUN:  [{key: cppcoreguidelines-no-malloc.Allocations, value: "::malloc"},\
// RUN:   {key: cppcoreguidelines-no-malloc.Reallocations, value: ""},\
// RUN:   {key: cppcoreguidelines-no-malloc.Deallocations, value: ""}]}' \
// RUN: --

// Just ensure, the check will not crash, when no functions shall be checked.

using size_t = __SIZE_TYPE__;

void *malloc(size_t size);

void malloced_array() {
  int *array0 = (int *)malloc(sizeof(int) * 20);
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: do not manage memory manually; consider a container or a smart pointer [cppcoreguidelines-no-malloc]
}
