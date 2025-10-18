// RUN: %check_clang_tidy -std=c++20 %s modernize-use-constexpr %t

namespace std {
template <typename T = void>
struct coroutine_handle {
   static constexpr coroutine_handle from_address(void* addr) {
     return {};
   }
};

struct always_suspend {
   bool await_ready() const noexcept;
   bool await_resume() const noexcept;
   template <typename T>
   bool await_suspend(coroutine_handle<T>) const noexcept;
};

template <typename T>
struct coroutine_traits {
   using promise_type = T::promise_type;
};
}  // namespace std

struct generator {
   struct promise_type {
       void return_value(int v);
       std::always_suspend yield_value(int&&);
       std::always_suspend initial_suspend() const noexcept;
       std::always_suspend final_suspend() const noexcept;
       void unhandled_exception();
       generator get_return_object();
   };
};


generator f25() { co_return 10; }
