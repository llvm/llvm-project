// RUN: %clang_cc1 -analyze -fblocks -std=c++11 -analyzer-checker=alpha.clone.CloneChecker -verify %s

// This tests if we search for clones in blocks.

void log();

auto BlockA = ^(int a, int b){ // expected-warning{{Detected code clone.}}
  log();
  if (a > b)
    return a;
  return b;
};

auto BlockB = ^(int a, int b){ // expected-note{{Related code clone is here.}}
  log();
  if (a > b)
    return a;
  return b;
};
