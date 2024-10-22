// RUN: %check_clang_tidy -expect-clang-tidy-error %s cppcoreguidelines-pro-type-member-init %t

struct X {
  X x;
  // CHECK-MESSAGES: :[[@LINE-1]]:5: error: field has incomplete type 'X' [clang-diagnostic-error]
  int a = 10;
};

template <typename T> class NoCrash {
  // CHECK-MESSAGES: :[[@LINE+2]]:20: error: base class has incomplete type
  // CHECK-MESSAGES: :[[@LINE-2]]:29: note: definition of 'NoCrash<T>' is not complete until the closing '}'
  class B : public NoCrash {
    template <typename U> B(U u) {}
  };
};
