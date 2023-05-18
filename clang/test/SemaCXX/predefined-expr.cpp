// RUN: %clang_cc1 -x c++ -std=c++1y -fblocks -fsyntax-only -triple %itanium_abi_triple -verify %s
// PR16946
// expected-no-diagnostics

auto foo() {
  static_assert(sizeof(__func__) == 4, "foo");
  static_assert(sizeof(__FUNCTION__) == 4, "foo");
  static_assert(sizeof(__PRETTY_FUNCTION__) == 11, "auto foo()");
  return 0;
}

auto bar() -> decltype(42) {
  static_assert(sizeof(__func__) == 4, "bar");
  static_assert(sizeof(__FUNCTION__) == 4, "bar");
  static_assert(sizeof(__PRETTY_FUNCTION__) == 10, "int bar()");
  return 0;
}

// Within templates.
template <typename T>
int baz() {
  static_assert(sizeof(__func__) == 4, "baz");
  static_assert(sizeof(__FUNCTION__) == 4, "baz");
  static_assert(sizeof(__PRETTY_FUNCTION__) == 20, "int baz() [T = int]");

  []() {
    static_assert(sizeof(__func__) == 11, "operator()");
    static_assert(sizeof(__FUNCTION__) == 11, "operator()");
    static_assert(sizeof(__PRETTY_FUNCTION__) == 50,
                  "auto baz()::<anonymous class>::operator()() const");
    return 0;
  }
  ();

  ^{
    static_assert(sizeof(__func__) == 17, "baz_block_invoke");
    static_assert(sizeof(__FUNCTION__) == 17, "baz_block_invoke");
    static_assert(sizeof(__PRETTY_FUNCTION__) == 33, "int baz() [T = int]_block_invoke");
  }();

  #pragma clang __debug captured
  {
    static_assert(sizeof(__func__) == 4, "baz");
    static_assert(sizeof(__FUNCTION__) == 4, "baz");
    static_assert(sizeof(__PRETTY_FUNCTION__) == 20, "int baz() [T = int]");
  }

  return 0;
}

int main() {
  static_assert(sizeof(__func__) == 5, "main");
  static_assert(sizeof(__FUNCTION__) == 5, "main");
  static_assert(sizeof(__PRETTY_FUNCTION__) == 11, "int main()");

  []() {
    static_assert(sizeof(__func__) == 11, "operator()");
    static_assert(sizeof(__FUNCTION__) == 11, "operator()");
    static_assert(sizeof(__PRETTY_FUNCTION__) == 51,
                  "auto main()::<anonymous class>::operator()() const");
    return 0;
  }
  ();

  ^{
    static_assert(sizeof(__func__) == 18, "main_block_invoke");
    static_assert(sizeof(__FUNCTION__) == 18, "main_block_invoke");
    static_assert(sizeof(__PRETTY_FUNCTION__) == 24, "int main()_block_invoke");
  }();

  #pragma clang __debug captured
  {
    static_assert(sizeof(__func__) == 5, "main");
    static_assert(sizeof(__FUNCTION__) == 5, "main");
    static_assert(sizeof(__PRETTY_FUNCTION__) == 11, "int main()");

    #pragma clang __debug captured
    {
      static_assert(sizeof(__func__) == 5, "main");
      static_assert(sizeof(__FUNCTION__) == 5, "main");
      static_assert(sizeof(__PRETTY_FUNCTION__) == 11, "int main()");
    }
  }

  []() {
    #pragma clang __debug captured
    {
      static_assert(sizeof(__func__) == 11, "operator()");
      static_assert(sizeof(__FUNCTION__) == 11, "operator()");
      static_assert(sizeof(__PRETTY_FUNCTION__) == 51,
                    "auto main()::<anonymous class>::operator()() const");
    }
  }
  ();

  baz<int>();

  return 0;
}
