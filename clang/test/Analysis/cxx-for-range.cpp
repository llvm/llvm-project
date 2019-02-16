// RUN: %clang_analyze_cc1 -std=c++11 -analyzer-checker=core -analyzer-output=plist-multi-file -o %t.plist -verify -analyzer-config eagerly-assume=false %s
// RUN: cat %t.plist | %diff_plist %S/Inputs/expected-plists/cxx-for-range.cpp.plist -

extern void work();

void testLoop() {
  int z[] = {1,2};
  for (int y : z) {
    work();
    work();
    if (y == 2)
      *(volatile int *)0 = 1; // expected-warning {{Dereference of null pointer}}
    work();
    work();
    (void)y;
  }

  *(volatile int *)0 = 1; // no-warning
}

class MagicVector {
public:
  MagicVector();

  using iterator = int *;

  iterator begin() const;
  iterator end() const;
};

MagicVector get(bool fail = false) {
  if (fail)
    *(volatile int *)0 = 1; // expected-warning {{Dereference of null pointer}}
  return MagicVector{};
}

void testLoopOpaqueCollection() {
  for (int y : get()) {
    work();
    work();
    if (y == 2)
      *(volatile int *)0 = 1; // expected-warning {{Dereference of null pointer}}
    work();
    work();
    (void)y;
  }

  *(volatile int *)0 = 1; // expected-warning {{Dereference of null pointer}}
}


class MagicVector2 {
public:
  MagicVector2();

  class iterator {
  public:
    int operator*() const;
    iterator &operator++();
    bool operator==(const iterator &);
    bool operator!=(const iterator &);
  };

  iterator begin() const;
  iterator end() const;
};

MagicVector2 get2() {
  return MagicVector2{};
}

void testLoopOpaqueIterator() {
  for (int y : get2()) {
    work();
    work();
    if (y == 2)
      *(volatile int *)0 = 1; // expected-warning {{Dereference of null pointer}}
    work();
    work();
    (void)y;
  }

  *(volatile int *)0 = 1; // expected-warning {{Dereference of null pointer}}
}


void testLoopErrorInRange() {
  for (int y : get(true)) { // error inside get()
    work();
    work();
    if (y == 2)
      *(volatile int *)0 = 1; // no-warning
    work();
    work();
    (void)y;
  }

  *(volatile int *)0 = 1; // no-warning
}

void testForRangeInit() {
  for (int *arr[3] = {nullptr, nullptr, nullptr}; int *p : arr) // expected-warning {{extension}}
    *p = 1; // expected-warning {{Dereference of null pointer}}
}
