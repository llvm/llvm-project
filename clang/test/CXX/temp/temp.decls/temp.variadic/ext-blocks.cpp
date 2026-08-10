// RUN: %clang_cc1 -std=c++11 -fblocks -fsyntax-only -verify %s

// Tests the use of blocks with variadic templates.
template<typename ...Args>
int f0(Args ...args) {
  return ^ {
    return sizeof...(Args);
  }() + ^ {
    return sizeof...(args);
  }();
}

template<typename ...Args>
int f1(Args ...args) {
  return ^ {
    return f0(args...);
  }();
}

template int f0(int, float, double);
template int f1(const char*, int, float, double);

template<typename ...Args>
int f2(Args ...args) {
  return ^(Args ...block_args) {
    return f1(block_args...);
  }(args + 0 ...);
}

template int f2(const char*, int, float, double);

template<typename ...Args>
int f3(Args ...args) {
  return ^(Args *...block_args) {
    return f1(block_args...);
  }(&args...);
}

template int f3(const char*, int, float, double);

template<typename ...Args>
int PR9953(Args ...args) {
  return ^(Args *...block_args) {
    return f1(block_args); // expected-error{{expression contains unexpanded parameter pack 'block_args'}}
  }(&args...);
}
