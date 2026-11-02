// RUN: %check_clang_tidy -std=c++23-or-later %s performance-inefficient-container-assignment %t

#include <vector>

void fromRange(std::vector<int> &V, const std::vector<int> &Other,
               const int (&Array)[3]) {
  V = std::vector<int>(std::from_range, Other);
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: inefficient assignment from a temporary 'std::vector'; use 'assign_range' to reuse the existing storage [performance-inefficient-container-assignment]
  // CHECK-FIXES: V.assign_range(Other);

  V = std::vector<int>(std::from_range, Array);
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: inefficient assignment from a temporary 'std::vector'; use 'assign_range'
  // CHECK-FIXES: V.assign_range(Array);
}

void fromRangeWithAllocator(std::vector<int> &V, const std::vector<int> &Other,
                            std::allocator<int> Alloc) {
  // An explicitly passed allocator has no counterpart in 'assign_range'.
  V = std::vector<int>(std::from_range, Other, Alloc);
}

struct NoCopyAssign {
  NoCopyAssign(const NoCopyAssign &);
  NoCopyAssign &operator=(const NoCopyAssign &) = delete;
};

void notAssignable(std::vector<NoCopyAssign> &V,
                   const std::vector<NoCopyAssign> &Other) {
  // 'assign_range' needs assignable elements.
  V = std::vector<NoCopyAssign>(std::from_range, Other);
}
