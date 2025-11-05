// RUN: %clang -std=c++17 %s 2>&1 | FileCheck %s --check-prefix=NO-TSAN --allow-empty
// RUN: %clang -std=c++17 -fsanitize=thread %s 2>&1 | FileCheck %s --check-prefix=WITH-TSAN

// WITH-TSAN: `std::atomic_thread_fence` is not supported with `-fsanitize=thread`
// NO-TSAN-NOT: `std::atomic_thread_fence` is not supported with `-fsanitize=thread`

#include <atomic>

int main() {
    std::atomic_thread_fence(std::memory_order::memory_order_relaxed);
}
