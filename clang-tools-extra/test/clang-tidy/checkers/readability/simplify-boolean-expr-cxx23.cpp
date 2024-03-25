// RUN: clang-tidy %s -checks='-*,readability-simplify-boolean-expr' -- -std=c++2b | count 0
template <bool Cond>
constexpr int testIf() {
  if consteval {
    if constexpr (Cond) {
      return 0;
    } else {
      return 1;
    }
  } else {
    return 2;
  }
}

constexpr bool testCompound() {
  if consteval {
    return true;
  }
  return false;
}

constexpr bool testCase(int I) {
  switch (I) {
    case 0: {
      if consteval {
        return true;
      }
      return false;
    }
    default: {
      if consteval {
        return false;
      }
      return true;
    }
  }
}
