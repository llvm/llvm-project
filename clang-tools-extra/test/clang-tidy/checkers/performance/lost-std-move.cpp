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
  // CHECK-FIXES:   f(std::move(ptr));
}

void f_rvalue_ref(std::shared_ptr<int>&& ptr)
{
  if (*ptr)
    f(ptr);
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: could be std::move() [performance-lost-std-move]
  // CHECK-FIXES:   f(std::move(ptr));
}

using SharedPtr = std::shared_ptr<int>;
void f_using(SharedPtr ptr)
{
  if (*ptr)
    f(ptr);
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: could be std::move() [performance-lost-std-move]
}

void f_thread_local()
{
  thread_local std::shared_ptr<int> ptr;
  if (*ptr)
    f(ptr);
}

void f_static()
{
  static std::shared_ptr<int> ptr;
  if (*ptr)
    f(ptr);
}

void f_extern()
{
  extern std::shared_ptr<int> ptr;
  if (*ptr)
    f(ptr);
}

std::shared_ptr<int> global;
void f_global()
{
  f(global);
}

void f_local()
{
  std::shared_ptr<int> ptr;
  if (*ptr)
    f(ptr);
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: could be std::move() [performance-lost-std-move]
  // CHECK-FIXES:   f(std::move(ptr));
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
  } while (true);
}

int f_multiple_usages()
{
  std::shared_ptr<int> ptr;
  return f(ptr) + f(ptr);
}

#define FUN(x) f((x))
int f_macro()
{
  std::shared_ptr<int> ptr;
  return FUN(ptr);
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: could be std::move() [performance-lost-std-move]
  // CHECK-FIXES: return FUN(std::move(ptr));
}

void f_lambda_ref()
{
  std::shared_ptr<int> ptr;
  auto Lambda = [&ptr]() mutable {
    f(ptr);
    };
  Lambda();
}

void f_lambda()
{
  std::shared_ptr<int> ptr;
  auto Lambda = [ptr]() mutable {
    // CHECK-MESSAGES: [[@LINE-1]]:18: warning: could be std::move() [performance-lost-std-move]
    // CHECK-FIXES: auto Lambda = [std::move(ptr)]() mutable {
    f(ptr);
    };
  Lambda();
}

void f_lambda_assign()
{
  std::shared_ptr<int> ptr;
  auto Lambda = [ptr = ptr]() mutable {
    // CHECK-MESSAGES: [[@LINE-1]]:24: warning: could be std::move() [performance-lost-std-move]
    // CHECK-FIXES: auto Lambda = [ptr = std::move(ptr)]() mutable {
    f(ptr);
    };
  Lambda();
}

void f_lambda_assign_all()
{
  std::shared_ptr<int> ptr;
  auto Lambda = [=]() mutable {
    // CHECK-MESSAGES: [[@LINE-1]]:18: warning: could be std::move() [performance-lost-std-move]
    // CHECK-FIXES: auto Lambda = [ptr = std::move(ptr),=]() mutable {
    f(ptr);
    };
  Lambda();
}

void f_copy_after_ref()
{
  std::shared_ptr<int> ptr;
  auto& ref = ptr;
  f(ptr);
  *ref = 1;
}

int f_lvalue(std::shared_ptr<int>&);
int f_lvalue_const(const std::shared_ptr<int>&);

void f_ref_lvalue(std::shared_ptr<int> ptr)
{
  f_lvalue(ptr); // no fix
}

void f_ref_lvalue_const(std::shared_ptr<int> ptr)
{
  f_lvalue_const(ptr); // no fix
}
