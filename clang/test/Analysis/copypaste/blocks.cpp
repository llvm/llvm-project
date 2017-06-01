// RUN: %clang_analyze_cc1 -fblocks -std=c++11 -analyzer-checker=alpha.clone.CloneChecker -verify %s

// This tests if we search for clones in blocks.

void log();

auto BlockA = ^(int a, int b){ // expected-warning{{Duplicate code detected}}
  log();
  if (a > b)
    return a;
  return b;
};

auto BlockB = ^(int a, int b){ // expected-note{{Similar code here}}
  log();
  if (a > b)
    return a;
  return b;
};
