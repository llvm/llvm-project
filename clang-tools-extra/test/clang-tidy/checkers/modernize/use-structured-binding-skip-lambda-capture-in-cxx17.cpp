// RUN: %check_clang_tidy -std=c++17 %s modernize-use-structured-binding %t -- -- -I %S/Inputs/use-structured-binding/

#include "fake_std_pair_tuple.h"

void captureByVal() {
  auto P = getPair<int, int>();
  int x = P.first;
  int y = P.second;

  auto lambda = [x]() {
    int y = x;
  };
}

void captureByRef() {
  auto P = getPair<int, int>();
  int x = P.first;
  int y = P.second;

  auto lambda = [&x]() {
    x = 1;
  };
}

void captureByAllRef() {
  auto P = getPair<int, int>();
  int x = P.first;
  int y = P.second;

  auto lambda = [&]() {
    x = 1;
  };
}

void deepLambda() {
  auto P = getPair<int, int>();
  int x = P.first;
  int y = P.second;

  {
    auto lambda = [x]() {
      int y = x;
    };
  }
}

void forRangeNotWarn() {
  std::pair<int, int> Pairs[10];
  for (auto P : Pairs) {
    int x = P.first;
    int y = P.second;

    auto lambda = [&]() {
    x = 1;
  };
  }
}

void stdTieNotWarn() {
  int x = 0;
  int y = 0;
  std::tie(x, y) = getPair<int, int>();

  auto lambda = [&x]() {
    x = 1;
  };
}
