// RUN: %check_clang_tidy %s performance-move-shared-ptr %t

#include <utility>

namespace std {
template<typename T>
class shared_ptr {
public:
  T& operator*() { return reinterpret_cast<T&>(*this); }
};

/*
template<typename T>
T&& move(T&)
{
}
*/

} // namespace std

void f(std::shared_ptr<int>);

void f_arg(std::shared_ptr<int> ptr)
{
  if (*ptr)
    f(ptr);
}

void f_local()
{
  std::shared_ptr<int> ptr;
  if (*ptr)
    f(ptr);
}

void f_move()
{
  std::shared_ptr<int> ptr;
  if (*ptr)
    f(std::move(ptr));
}

void f_return()
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
