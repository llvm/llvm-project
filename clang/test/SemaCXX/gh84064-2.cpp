// RUN: %clang_cc1 -fsyntax-only -verify -I%S/Inputs -std=c++23 %s

// expected-no-diagnostics

#include "std-coroutine.h"

using size_t = decltype(sizeof(0));

struct GeneratorStatic {
  struct promise_type {
    int _val{};

    GeneratorStatic get_return_object() noexcept
    {
      return {};
    }

    std::suspend_never initial_suspend() noexcept
    {
      return {};
    }

    std::suspend_always final_suspend() noexcept
    {
      return {};
    }

    void return_void() noexcept {}
    void unhandled_exception() noexcept {}

    template<typename... TheRest>
    static void*
    operator new(size_t  size,
                 TheRest&&...) noexcept
    {
        return nullptr;
    }

    static void operator delete(void*, size_t)
    {
    }
  };
};


int main()
{
  auto lambCpp23 = []() static -> GeneratorStatic {
    co_return;
  };

  static_assert(sizeof(decltype(lambCpp23)) == 1);
}
