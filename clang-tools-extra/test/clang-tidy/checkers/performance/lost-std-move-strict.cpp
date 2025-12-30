// RUN: %check_clang_tidy %s performance-lost-std-move %t -- -config='{CheckOptions: {performance-lost-std-move.StrictMode: true}}'

namespace std {

template<typename T>
class shared_ptr {
public:
  T& operator*() { return reinterpret_cast<T&>(*this); }
  shared_ptr() {}
  shared_ptr(const shared_ptr<T>&) {}
};

template<typename T>
T&& move(T&)
{
}

} // namespace std

int f(std::shared_ptr<int>);

void f_copy_after_ref()
{
  std::shared_ptr<int> ptr;
  auto& ref = ptr;
  f(ptr);
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: could be std::move() [performance-lost-std-move]
  // CHECK-FIXES: f(std::move(ptr));
  *ref = 1;
}
