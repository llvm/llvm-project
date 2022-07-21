//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

#include <algorithm>
#include <cassert>
#include <vector>

struct Cpp17ProxyIterator {
  struct Reference {
    int* i_;
    Reference(int& i) : i_(&i) {}

    operator int() const { return *i_; }

    Reference& operator=(int i) {
      *i_ = i;
      return *this;
    }

    friend bool operator<(const Reference& x, const Reference& y) { return *x.i_ < *y.i_; }

    friend bool operator==(const Reference& x, const Reference& y) { return *x.i_ == *y.i_; }

    friend void swap(Reference x, Reference y) { std::swap(*(x.i_), *(y.i_)); }
  };

  using difference_type   = int;
  using value_type        = int;
  using reference         = Reference;
  using pointer           = void*;
  using iterator_category = std::random_access_iterator_tag;

  int* ptr_;

  Cpp17ProxyIterator(int* ptr) : ptr_(ptr) {}

  Reference operator*() const { return Reference(*ptr_); }

  Cpp17ProxyIterator& operator++() {
    ++ptr_;
    return *this;
  }

  Cpp17ProxyIterator operator++(int) {
    auto tmp = *this;
    ++*this;
    return tmp;
  }

  friend bool operator==(const Cpp17ProxyIterator& x, const Cpp17ProxyIterator& y) { return x.ptr_ == y.ptr_; }
  friend bool operator!=(const Cpp17ProxyIterator& x, const Cpp17ProxyIterator& y) { return x.ptr_ != y.ptr_; }

  Cpp17ProxyIterator& operator--() {
    --ptr_;
    return *this;
  }

  Cpp17ProxyIterator operator--(int) {
    auto tmp = *this;
    --*this;
    return tmp;
  }

  Cpp17ProxyIterator& operator+=(difference_type n) {
    ptr_ += n;
    return *this;
  }

  Cpp17ProxyIterator& operator-=(difference_type n) {
    ptr_ -= n;
    return *this;
  }

  Reference operator[](difference_type i) const { return Reference(*(ptr_ + i)); }

  friend bool operator<(const Cpp17ProxyIterator& x, const Cpp17ProxyIterator& y) { return x.ptr_ < y.ptr_; }

  friend bool operator>(const Cpp17ProxyIterator& x, const Cpp17ProxyIterator& y) { return x.ptr_ > y.ptr_; }

  friend bool operator<=(const Cpp17ProxyIterator& x, const Cpp17ProxyIterator& y) { return x.ptr_ <= y.ptr_; }

  friend bool operator>=(const Cpp17ProxyIterator& x, const Cpp17ProxyIterator& y) { return x.ptr_ >= y.ptr_; }

  friend Cpp17ProxyIterator operator+(const Cpp17ProxyIterator& x, difference_type n) {
    return Cpp17ProxyIterator(x.ptr_ + n);
  }

  friend Cpp17ProxyIterator operator+(difference_type n, const Cpp17ProxyIterator& x) {
    return Cpp17ProxyIterator(n + x.ptr_);
  }

  friend Cpp17ProxyIterator operator-(const Cpp17ProxyIterator& x, difference_type n) {
    return Cpp17ProxyIterator(x.ptr_ - n);
  }

  friend difference_type operator-(Cpp17ProxyIterator x, Cpp17ProxyIterator y) {
    return static_cast<int>(x.ptr_ - y.ptr_);
  }
};

void test() {
  // TODO: use a custom proxy iterator instead of (or in addition to) `vector<bool>`.
  std::vector<bool> v(5, false);
  v[1] = true;
  v[3] = true;
  std::sort(v.begin(), v.end());
  assert(std::is_sorted(v.begin(), v.end()));
}

void testCustomProxyIterator() {
  int a[] = {5, 1, 3, 2, 4};
  std::sort(Cpp17ProxyIterator(a), Cpp17ProxyIterator(a + 5));
  assert(a[0] == 1);
  assert(a[1] == 2);
  assert(a[2] == 3);
  assert(a[3] == 4);
  assert(a[4] == 5);
}

int main(int, char**) {
  test();
  testCustomProxyIterator();
  return 0;
}
