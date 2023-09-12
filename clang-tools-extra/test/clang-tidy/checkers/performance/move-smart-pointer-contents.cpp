// RUN: %check_clang_tidy %s performance-move-smart-pointer-contents %t -- \
// RUN: -config="{CheckOptions: \
// RUN:           {performance-move-smart-pointer-contents.UniquePointerClasses: \
// RUN:              'std::unique_ptr; my::OtherUniquePtr;'}}"

// Some dummy definitions we'll need.

namespace std {

using size_t = int;
  
template <typename> struct remove_reference;
template <typename _Tp> struct remove_reference { typedef _Tp type; };
template <typename _Tp> struct remove_reference<_Tp &> { typedef _Tp type; };
template <typename _Tp> struct remove_reference<_Tp &&> { typedef _Tp type; };

template <typename _Tp>
constexpr typename std::remove_reference<_Tp>::type &&move(_Tp &&__t) {
  return static_cast<typename std::remove_reference<_Tp>::type &&>(__t);
}

template <typename T>
struct unique_ptr {
  unique_ptr();
  T *get() const;
  explicit operator bool() const;
  void reset(T *ptr);
  T &operator*() const;
  T *operator->() const;
  T& operator[](size_t i) const;
};
  
}  // namespace std

namespace my {
template <typename T>
using OtherUniquePtr = std::unique_ptr<T>;
}  // namespace my

void correctUnique() {
  std::unique_ptr<int> p;
  int x = *std::move(p);
}

void simpleFindingUnique() {
  std::unique_ptr<int> p;
  int x = std::move(*p);
}
// CHECK-MESSAGES: :[[@LINE-2]]:11: warning: prefer to move the smart pointer rather than its contents [performance-move-smart-pointer-contents]

void aliasedUniqueType() {
  using int_ptr = std::unique_ptr<int>;
  int_ptr p;
  int x = std::move(*p);
}
// CHECK-MESSAGES: :[[@LINE-2]]:11: warning: prefer to move the smart pointer rather than its contents [performance-move-smart-pointer-contents]

void configWorksUnique() {
  my::OtherUniquePtr<int> p;
  int x = std::move(*p);
}
// CHECK-MESSAGES: :[[@LINE-2]]:11: warning: prefer to move the smart pointer rather than its contents [performance-move-smart-pointer-contents]

void multiStarsUnique() {
  std::unique_ptr<int> p;
  int x = 2 * std::move(*p) * 3;
}
// CHECK-MESSAGES: :[[@LINE-2]]:15: warning: prefer to move the smart pointer rather than its contents [performance-move-smart-pointer-contents]
