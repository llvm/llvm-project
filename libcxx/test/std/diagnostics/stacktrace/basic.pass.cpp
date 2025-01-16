//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

#include <experimental/stacktrace>
#include <iterator>

#include <cassert>
#include <type_traits>

/*
  (19.6.4) Class template basic_stacktrace [stacktrace.basic]
  (19.6.4.1) Overview                      [stacktrace.basic.overview]

namespace std {
  template<class Allocator>
  class basic_stacktrace {
  public:
    using value_type = stacktrace_entry;
    using const_reference = const value_type&;
    using reference = value_type&;
    using const_iterator = implementation-defined;  // see [stacktrace.basic.obs]
    using iterator = const_iterator;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;
    using difference_type = implementation-defined;
    using size_type = implementation-defined;
    using allocator_type = Allocator;

    // (19.6.4.2)
    // [stacktrace.basic.cons], creation and assignment
    static basic_stacktrace current(const allocator_type& alloc = allocator_type()) noexcept;
    static basic_stacktrace current(size_type skip,
                                    const allocator_type& alloc = allocator_type()) noexcept;
    static basic_stacktrace current(size_type skip, size_type max_depth,
                                    const allocator_type& alloc = allocator_type()) noexcept;

    basic_stacktrace() noexcept(is_nothrow_default_constructible_v<allocator_type>);
    explicit basic_stacktrace(const allocator_type& alloc) noexcept;

    basic_stacktrace(const basic_stacktrace& other);
    basic_stacktrace(basic_stacktrace&& other) noexcept;
    basic_stacktrace(const basic_stacktrace& other, const allocator_type& alloc);
    basic_stacktrace(basic_stacktrace&& other, const allocator_type& alloc);
    basic_stacktrace& operator=(const basic_stacktrace& other);
    basic_stacktrace& operator=(basic_stacktrace&& other)
      noexcept(allocator_traits<Allocator>::propagate_on_container_move_assignment::value ||
        allocator_traits<Allocator>::is_always_equal::value);

    ~basic_stacktrace();

    // (19.6.4.3)
    // [stacktrace.basic.obs], observers
    allocator_type get_allocator() const noexcept;

    const_iterator begin() const noexcept;
    const_iterator end() const noexcept;
    const_reverse_iterator rbegin() const noexcept;
    const_reverse_iterator rend() const noexcept;

    const_iterator cbegin() const noexcept;
    const_iterator cend() const noexcept;
    const_reverse_iterator crbegin() const noexcept;
    const_reverse_iterator crend() const noexcept;

    bool empty() const noexcept;
    size_type size() const noexcept;
    size_type max_size() const noexcept;

    const_reference operator[](size_type) const;
    const_reference at(size_type) const;
    
    // (19.6.4.4)
    // [stacktrace.basic.cmp], comparisons
    template<class Allocator2>
    friend bool operator==(const basic_stacktrace& x,
                           const basic_stacktrace<Allocator2>& y) noexcept;
    template<class Allocator2>
    friend strong_ordering operator<=>(const basic_stacktrace& x,
                                       const basic_stacktrace<Allocator2>& y) noexcept;

    // (19.6.4.5)
    // [stacktrace.basic.mod], modifiers
    void swap(basic_stacktrace& other)
      noexcept(allocator_traits<Allocator>::propagate_on_container_swap::value ||
        allocator_traits<Allocator>::is_always_equal::value);

  private:
    vector<value_type, allocator_type> frames_;         // exposition only
  };

  // (19.6.4.6)
  // [stacktrace.basic.nonmem], non-member functions

  template<class Allocator>
    void swap(basic_stacktrace<Allocator>& a, basic_stacktrace<Allocator>& b)
      noexcept(noexcept(a.swap(b)));

  string to_string(const stacktrace_entry& f);

  template<class Allocator>
    string to_string(const basic_stacktrace<Allocator>& st);

  ostream& operator<<(ostream& os, const stacktrace_entry& f);
  template<class Allocator>
    ostream& operator<<(ostream& os, const basic_stacktrace<Allocator>& st);
}

*/

int main(int, char**) {
  // using value_type = stacktrace_entry;
  // using const_reference = const value_type&;
  // using reference = value_type&;
  // using const_iterator = implementation-defined;  // see [stacktrace.basic.obs]
  // using iterator = const_iterator;
  // using reverse_iterator = std::reverse_iterator<iterator>;
  // using const_reverse_iterator = std::reverse_iterator<const_iterator>;
  // using difference_type = implementation-defined;
  // using size_type = implementation-defined;
  // using allocator_type = Allocator;

  // This test will only verify these member types exist and are
  // defined as expected; their actual behavior is tested in another .cpp.

  using A = std::allocator<std::stacktrace_entry>;
  using S = std::basic_stacktrace<A>;

  static_assert(std::is_same_v<S::value_type, std::stacktrace_entry>);
  static_assert(std::is_same_v<S::const_reference, std::stacktrace_entry const&>);
  static_assert(std::is_same_v<S::reference, std::stacktrace_entry&>);

  static_assert(std::forward_iterator<S::const_iterator>);
  static_assert(std::forward_iterator<S::iterator>);
  using CRI = S::const_reverse_iterator;
  static_assert(std::is_same_v<CRI, decltype(S(A()).crbegin())>);
  using RI = S::reverse_iterator;
  static_assert(std::is_same_v<RI, decltype(S(A()).rbegin())>);

  using IterT = S::iterator;
  using DiffT = S::difference_type;
  static_assert(std::is_same_v<IterT, decltype(IterT() + DiffT())>);

  static_assert(std::is_integral_v<S::size_type>);
  static_assert(std::is_same_v<S::allocator_type, A>);

  return 0;
}
