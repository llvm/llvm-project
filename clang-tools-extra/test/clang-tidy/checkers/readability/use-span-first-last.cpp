// RUN: %check_clang_tidy -std=c++20 %s readability-use-span-first-last %t

namespace std {

enum class byte : unsigned char {};

template <typename T>
class span {
  T* ptr;
  __SIZE_TYPE__ len;

public:
  span(T* p, __SIZE_TYPE__ l) : ptr(p), len(l) {}
  
  span<T> subspan(__SIZE_TYPE__ offset) const {
    return span(ptr + offset, len - offset);
  }
  
  span<T> subspan(__SIZE_TYPE__ offset, __SIZE_TYPE__ count) const {
    return span(ptr + offset, count);
  }

  span<T> first(__SIZE_TYPE__ count) const {
    return span(ptr, count);
  }

  span<T> last(__SIZE_TYPE__ count) const {
    return span(ptr + (len - count), count);
  }

  __SIZE_TYPE__ size() const { return len; }
  __SIZE_TYPE__ size_bytes() const { return len * sizeof(T); }
};
} // namespace std

// Add here, right after the std namespace closes:
namespace std::ranges {
  template<typename T>
  __SIZE_TYPE__ size(const span<T>& s) { return s.size(); }
}

void test() {
  int arr[] = {1, 2, 3, 4, 5};
  std::span<int> s(arr, 5);

  auto sub1 = s.subspan(0, 3);
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: prefer 'span::first()' over 'subspan()'
  // CHECK-FIXES: auto sub1 = s.first(3);

  auto sub2 = s.subspan(s.size() - 2);
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: prefer 'span::last()' over 'subspan()'
  // CHECK-FIXES: auto sub2 = s.last(2);

  __SIZE_TYPE__ n = 2;
  auto sub3 = s.subspan(0, n);
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: prefer 'span::first()' over 'subspan()'
  // CHECK-FIXES: auto sub3 = s.first(n);

  auto sub4 = s.subspan(1, 2);  // No warning
  auto sub5 = s.subspan(2);     // No warning


#define ZERO 0
#define TWO 2
#define SIZE_MINUS(s, n) s.size() - n
#define MAKE_SUBSPAN(obj, n) obj.subspan(0, n)
#define MAKE_LAST_N(obj, n) obj.subspan(obj.size() - n)

  auto sub6 = s.subspan(SIZE_MINUS(s, 2));
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: prefer 'span::last()' over 'subspan()'
  // CHECK-FIXES: auto sub6 = s.last(2);

  auto sub7 = MAKE_SUBSPAN(s, 3);
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: prefer 'span::first()' over 'subspan()'
  // CHECK-FIXES: auto sub7 = s.first(3);

  auto sub8 = MAKE_LAST_N(s, 2);
  // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: prefer 'span::last()' over 'subspan()'
  // CHECK-FIXES: auto sub8 = s.last(2);

}

template <typename T>
void testTemplate() {
  T arr[] = {1, 2, 3, 4, 5};
  std::span<T> s(arr, 5);

  auto sub1 = s.subspan(0, 3);
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: prefer 'span::first()' over 'subspan()'
  // CHECK-FIXES: auto sub1 = s.first(3);

  auto sub2 = s.subspan(s.size() - 2);
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: prefer 'span::last()' over 'subspan()'
  // CHECK-FIXES: auto sub2 = s.last(2);

  __SIZE_TYPE__ n = 2;
  auto sub3 = s.subspan(0, n);
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: prefer 'span::first()' over 'subspan()'
  // CHECK-FIXES: auto sub3 = s.first(n);

  auto sub4 = s.subspan(1, 2);  // No warning
  auto sub5 = s.subspan(2);     // No warning

  auto complex = s.subspan(0 + (s.size() - 2), 3);  // No warning

  auto complex2 = s.subspan(100 + (s.size() - 2));  // No warning
}

// Test instantiation
void testInt() {
  testTemplate<int>();
}

void test_ranges() {
  int arr[] = {1, 2, 3, 4, 5};
  std::span<int> s(arr, 5);

  auto sub1 = s.subspan(std::ranges::size(s) - 2);
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: prefer 'span::last()' over 'subspan()'
  // CHECK-FIXES: auto sub1 = s.last(2);

  __SIZE_TYPE__ n = 2;
  auto sub2 = s.subspan(std::ranges::size(s) - n);
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: prefer 'span::last()' over 'subspan()'
  // CHECK-FIXES: auto sub2 = s.last(n);
}

void test_different_spans() {
  int arr1[] = {1, 2, 3, 4, 5};
  int arr2[] = {6, 7, 8, 9, 10};
  std::span<int> s1(arr1, 5);
  std::span<int> s2(arr2, 5);

  // These should NOT trigger warnings as they use size() from a different span
  auto sub1 = s1.subspan(s2.size() - 2);     // No warning
  auto sub2 = s2.subspan(s1.size() - 3);     // No warning
  
  // Also check with std::ranges::size
  auto sub3 = s1.subspan(std::ranges::size(s2) - 2);  // No warning
  auto sub4 = s2.subspan(std::ranges::size(s1) - 3);  // No warning

  // Mixed usage should also not trigger
  auto sub5 = s1.subspan(s2.size() - s1.size());      // No warning
  
  // Verify that correct usage still triggers warnings
  auto good1 = s1.subspan(s1.size() - 2);
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: prefer 'span::last()' over 'subspan()'
  // CHECK-FIXES: auto good1 = s1.last(2);
  
  auto good2 = s2.subspan(std::ranges::size(s2) - 3);
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: prefer 'span::last()' over 'subspan()'
  // CHECK-FIXES: auto good2 = s2.last(3);
}

void test_span_of_bytes() {
  std::byte arr[] = {std::byte{0x1}, std::byte{0x2}, std::byte{0x3},
                     std::byte{0x4}, std::byte{0x5}};
  std::span<std::byte> s(arr, 5);

  auto sub1 = s.subspan(0, 3);
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: prefer 'span::first()' over 'subspan()'
  // CHECK-FIXES: auto sub1 = s.first(3);

  auto sub2 = s.subspan(s.size() - 2);
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: prefer 'span::last()' over 'subspan()'
  // CHECK-FIXES: auto sub2 = s.last(2);

  // size_bytes() is not the same as size() in general, so should not trigger
  auto sub3 = s.subspan(s.size_bytes() - 2);  // No warning
}

// Test uninstantiated template -- should still warn on dependent code
template <typename T>
void uninstantiated_template(std::span<T> s) {
  auto sub1 = s.subspan(0, 3);
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: prefer 'span::first()' over 'subspan()'
  // CHECK-FIXES: auto sub1 = s.first(3);

  auto sub2 = s.subspan(s.size() - 2);
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: prefer 'span::last()' over 'subspan()'
  // CHECK-FIXES: auto sub2 = s.last(2);

  auto sub3 = s.subspan(1, 2);  // No warning
  auto sub4 = s.subspan(2);     // No warning
}

