// RUN: %clang_cc1 -fsyntax-only -verify -I%S/Inputs -std=c++20 %s

// expected-no-diagnostics

#include "std-coroutine.h"

using size_t = decltype(sizeof(0));

struct Generator {
  struct promise_type {
    int _val{};

    Generator get_return_object() noexcept
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

    template<typename This, typename... TheRest>
    promise_type(This&,
                 TheRest&&...)
    {
    }
  };
};

struct CapturingThisTest
{
    int x{};

    void AsPointer()
    {
      auto lamb = [=,this]() -> Generator {
        int y = x;
        co_return;
      };

      static_assert(sizeof(decltype(lamb)) == sizeof(void*));
    }

    void AsStarThis()
    {
      auto lamb = [*this]() -> Generator {
        int y = x;
        co_return;
      };

      static_assert(sizeof(decltype(lamb)) == sizeof(int));
    }
};

int main()
{
  auto lamb = []() -> Generator {
    co_return;
  };

  static_assert(sizeof(decltype(lamb)) == 1);
}
