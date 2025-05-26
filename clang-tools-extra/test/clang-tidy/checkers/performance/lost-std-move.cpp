// RUN: %check_clang_tidy %s performance-lost-std-move %t

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

void f_arg(std::shared_ptr<int> ptr)
{
  if (*ptr)
    f(ptr);
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: could be std::move() [performance-lost-std-move]
}

void f_rvalue_ref(std::shared_ptr<int>&& ptr)
{
  if (*ptr)
    f(ptr);
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: could be std::move() [performance-lost-std-move]
}

using SharedPtr = std::shared_ptr<int>;
void f_using(SharedPtr ptr)
{
  if (*ptr)
    f(ptr);
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: could be std::move() [performance-lost-std-move]
}

void f_local()
{
  std::shared_ptr<int> ptr;
  if (*ptr)
    f(ptr);
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: could be std::move() [performance-lost-std-move]
}

void f_move()
{
  std::shared_ptr<int> ptr;
  if (*ptr)
    f(std::move(ptr));
}

void f_ref(std::shared_ptr<int> &ptr)
{
  if (*ptr)
    f(ptr);
}

std::shared_ptr<int> f_return()
{
  std::shared_ptr<int> ptr;
  return ptr;
}

void f_still_used(std::shared_ptr<int> ptr)
{
  if (*ptr)
    f(ptr);

  *ptr = 1;
  *ptr = *ptr;
}

void f_cycle1()
{
  std::shared_ptr<int> ptr;
  for(;;)
    f(ptr);
}

void f_cycle2()
{
  std::shared_ptr<int> ptr;
  for(int i=0; i<5; i++)
    f(ptr);
}

void f_cycle3()
{
  std::shared_ptr<int> ptr;
  while (*ptr) {
    f(ptr);
  }
}

void f_cycle4()
{
  std::shared_ptr<int> ptr;
  do {
    f(ptr);
  } while (*ptr);
}

int f_multiple_usages()
{
  std::shared_ptr<int> ptr;
  return f(ptr) + f(ptr);
}
