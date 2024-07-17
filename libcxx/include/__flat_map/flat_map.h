// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef _LIBCPP___FLAT_MAP_FLAT_MAP_H
#define _LIBCPP___FLAT_MAP_FLAT_MAP_H

#include <__algorithm/ranges_equal.h>
#include <__algorithm/ranges_lexicographical_compare.h>
#include <__algorithm/ranges_lower_bound.h>
#include <__algorithm/ranges_stable_sort.h>
#include <__algorithm/ranges_unique.h>
#include <__algorithm/ranges_upper_bound.h>
#include <__compare/synth_three_way.h>
#include <__concepts/convertible_to.h>
#include <__config>
#include <__flat_map/sorted_unique.h>
#include <__functional/is_transparent.h>
#include <__functional/operations.h>
#include <__iterator/concepts.h>
#include <__iterator/distance.h>
#include <__iterator/iterator_traits.h>
#include <__iterator/ranges_iterator_traits.h>
#include <__iterator/reverse_iterator.h>
#include <__memory/allocator_traits.h>
#include <__memory/uses_allocator.h>
#include <__memory/uses_allocator_construction.h>
#include <__ranges/concepts.h>
#include <__ranges/container_compatible_range.h>
#include <__ranges/ref_view.h>
#include <__ranges/subrange.h>
#include <__ranges/zip_view.h>
#include <__type_traits/conjunction.h>
#include <__type_traits/invoke.h>
#include <__type_traits/is_allocator.h>
#include <__type_traits/is_nothrow_default_constructible.h>
#include <__type_traits/maybe_const.h>
#include <__utility/pair.h>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#if _LIBCPP_STD_VER >= 23

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Key,
          class _Tp,
          class _Compare         = less<_Key>,
          class _KeyContainer    = vector<_Key>,
          class _MappedContainer = vector<_Tp>>
class flat_map {
  template <bool _Const>
  struct __iterator;

  template <class, class, class, class, class>
  friend class flat_map;

public:
  // types
  using key_type    = _Key;
  using mapped_type = _Tp;
  using value_type  = pair<key_type, mapped_type>;
  using key_compare = _Compare;
  // TODO : the following is the spec, but not implementable for vector<bool>
  // using reference              = pair<const key_type&, mapped_type&>;
  // using const_reference        = pair<const key_type&, const mapped_type&>;
  using reference = pair<ranges::range_reference_t<const _KeyContainer>, ranges::range_reference_t<_MappedContainer>>;
  using const_reference =
      pair<ranges::range_reference_t<const _KeyContainer>, ranges::range_reference_t<const _MappedContainer>>;
  using size_type              = size_t;
  using difference_type        = ptrdiff_t;
  using iterator               = __iterator<false>; // see [container.requirements]
  using const_iterator         = __iterator<true>;  // see [container.requirements]
  using reverse_iterator       = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;
  using key_container_type     = _KeyContainer;
  using mapped_container_type  = _MappedContainer;

  class value_compare {
  private:
    key_compare __comp_;
    value_compare(key_compare __c) : __comp_(__c) {}
    friend flat_map;

  public:
    _LIBCPP_HIDE_FROM_ABI bool operator()(const_reference __x, const_reference __y) const {
      return __comp_(__x.first, __y.first);
    }
  };

  struct containers {
    key_container_type keys;
    mapped_container_type values;
  };

private:
  template <class _Allocator>
  _LIBCPP_HIDE_FROM_ABI static constexpr bool __allocator_ctor_constraint =
      _And<uses_allocator<key_container_type, _Allocator>, uses_allocator<mapped_container_type, _Allocator>>::value;

  _LIBCPP_HIDE_FROM_ABI static constexpr bool __is_compare_transparent = __is_transparent_v<_Compare, _Compare>;

  template <bool _Const>
  struct __iterator {
  private:
    using __key_iterator    = ranges::iterator_t<__maybe_const<_Const, key_container_type>>;
    using __mapped_iterator = ranges::iterator_t<__maybe_const<_Const, mapped_container_type>>;
    using __reference       = pair<iter_reference_t<__key_iterator>, iter_reference_t<__mapped_iterator>>;

    struct __arrow_proxy {
      __reference __ref_;
      _LIBCPP_HIDE_FROM_ABI __reference* operator->() { return std::addressof(__ref_); }
    };

    __key_iterator __key_iter_;
    __mapped_iterator __mapped_iter_;

    friend flat_map;

  public:
    using iterator_concept  = random_access_iterator_tag;
    using iterator_category = input_iterator_tag;
    using value_type        = flat_map::value_type;
    using difference_type   = flat_map::difference_type;

    _LIBCPP_HIDE_FROM_ABI __iterator() = default;

    _LIBCPP_HIDE_FROM_ABI __iterator(__iterator<!_Const> __i)
      requires _Const && convertible_to<ranges::iterator_t<key_container_type>, __key_iterator> &&
                   convertible_to<ranges::iterator_t<mapped_container_type>, __mapped_iterator>
        : __key_iter_(std::move(__i.__key_iter_)), __mapped_iter_(std::move(__i.__mapped_iter_)) {}

    _LIBCPP_HIDE_FROM_ABI __iterator(__key_iterator __key_iter, __mapped_iterator __mapped_iter)
        : __key_iter_(std::move(__key_iter)), __mapped_iter_(std::move(__mapped_iter)) {}

    _LIBCPP_HIDE_FROM_ABI __reference operator*() const { return __reference(*__key_iter_, *__mapped_iter_); }
    _LIBCPP_HIDE_FROM_ABI __arrow_proxy operator->() const { return __arrow_proxy(**this); }

    _LIBCPP_HIDE_FROM_ABI __iterator& operator++() {
      ++__key_iter_;
      ++__mapped_iter_;
      return *this;
    }

    _LIBCPP_HIDE_FROM_ABI __iterator operator++(int) {
      __iterator __tmp(*this);
      ++*this;
      return __tmp;
    }

    _LIBCPP_HIDE_FROM_ABI __iterator& operator--() {
      --__key_iter_;
      --__mapped_iter_;
      return *this;
    }

    _LIBCPP_HIDE_FROM_ABI __iterator operator--(int) {
      __iterator __tmp(*this);
      --*this;
      return __tmp;
    }

    _LIBCPP_HIDE_FROM_ABI __iterator& operator+=(difference_type __x) {
      __key_iter_ += __x;
      __mapped_iter_ += __x;
      return *this;
    }

    _LIBCPP_HIDE_FROM_ABI __iterator& operator-=(difference_type __x) {
      __key_iter_ += __x;
      __mapped_iter_ += __x;
      return *this;
    }

    _LIBCPP_HIDE_FROM_ABI __reference operator[](difference_type __n) const { return *(*this + __n); }

    _LIBCPP_HIDE_FROM_ABI friend constexpr bool operator==(const __iterator& __x, const __iterator& __y) {
      return __x.__key_iter_ == __y.__key_iter_;
    }

    _LIBCPP_HIDE_FROM_ABI friend bool operator<(const __iterator& __x, const __iterator& __y) {
      return __x.__key_iter_ < __y.__key_iter_;
    }

    _LIBCPP_HIDE_FROM_ABI friend bool operator>(const __iterator& __x, const __iterator& __y) { return __y < __x; }

    _LIBCPP_HIDE_FROM_ABI friend bool operator<=(const __iterator& __x, const __iterator& __y) { return !(__y < __x); }

    _LIBCPP_HIDE_FROM_ABI friend bool operator>=(const __iterator& __x, const __iterator& __y) { return !(__x < __y); }

    _LIBCPP_HIDE_FROM_ABI friend auto operator<=>(const __iterator& __x, const __iterator& __y)
      requires three_way_comparable<__key_iterator>
    {
      return __x.__key_iter_ <=> __y.__key_iter_;
    }

    _LIBCPP_HIDE_FROM_ABI friend __iterator operator+(const __iterator& __i, difference_type __n) {
      auto __tmp = __i;
      __tmp += __n;
      return __tmp;
    }

    _LIBCPP_HIDE_FROM_ABI friend __iterator operator+(difference_type __n, const __iterator& __i) { return __i + __n; }

    _LIBCPP_HIDE_FROM_ABI friend __iterator operator-(const __iterator& __i, difference_type __n) {
      auto __tmp = __i;
      __tmp -= __n;
      return __tmp;
    }

    _LIBCPP_HIDE_FROM_ABI friend difference_type operator-(const __iterator& __x, const __iterator& __y) {
      return difference_type(__x.__key_iter_ - __y.__key_iter_);
    }
  };

public:
  // [flat.map.cons], construct/copy/destroy
  _LIBCPP_HIDE_FROM_ABI flat_map() noexcept(
      is_nothrow_default_constructible_v<_KeyContainer> && is_nothrow_default_constructible_v<_MappedContainer> &&
      is_nothrow_default_constructible_v<_Compare>)
      : __containers_(), __compare_() {}

  template <class _Allocator>
    requires __allocator_ctor_constraint<_Allocator>
  _LIBCPP_HIDE_FROM_ABI flat_map(const flat_map& __other, const _Allocator& __alloc)
      : flat_map(__ctor_uses_allocator_tag{},
                 __alloc,
                 __other.__containers_.keys,
                 __other.__containers_.values,
                 __other.__compare_) {}

  template <class _Allocator>
    requires __allocator_ctor_constraint<_Allocator>
  _LIBCPP_HIDE_FROM_ABI flat_map(flat_map&& __other, const _Allocator& __alloc)
      : flat_map(__ctor_uses_allocator_tag{},
                 __alloc,
                 std::move(__other.__containers_.keys),
                 std::move(__other.__containers_.values),
                 std::move(__other.__compare_)) {}

  _LIBCPP_HIDE_FROM_ABI flat_map(
      key_container_type __key_cont, mapped_container_type __mapped_cont, const key_compare& __comp = key_compare())
      : __containers_{.keys = std::move(__key_cont), .values = std::move(__mapped_cont)}, __compare_(__comp) {
    __sort_and_unique();
  }

  template <class _Allocator>
    requires __allocator_ctor_constraint<_Allocator>
  _LIBCPP_HIDE_FROM_ABI
  flat_map(const key_container_type& __key_cont, const mapped_container_type& __mapped_cont, const _Allocator& __alloc)
      : flat_map(__ctor_uses_allocator_tag{}, __alloc, __key_cont, __mapped_cont) {
    __sort_and_unique();
  }

  template <class _Allocator>
    requires __allocator_ctor_constraint<_Allocator>
  _LIBCPP_HIDE_FROM_ABI
  flat_map(const key_container_type& __key_cont,
           const mapped_container_type& __mapped_cont,
           const key_compare& __comp,
           const _Allocator& __alloc)
      : flat_map(__ctor_uses_allocator_tag{}, __alloc, __key_cont, __mapped_cont, __comp) {
    __sort_and_unique();
  }

  _LIBCPP_HIDE_FROM_ABI
  flat_map(sorted_unique_t,
           key_container_type __key_cont,
           mapped_container_type __mapped_cont,
           const key_compare& __comp = key_compare())
      : __containers_{.keys = std::move(__key_cont), .values = std::move(__mapped_cont)}, __compare_(__comp) {}

  template <class _Allocator>
    requires __allocator_ctor_constraint<_Allocator>
  _LIBCPP_HIDE_FROM_ABI
  flat_map(sorted_unique_t,
           const key_container_type& __key_cont,
           const mapped_container_type& __mapped_cont,
           const _Allocator& __alloc)
      : flat_map(__ctor_uses_allocator_tag{}, __alloc, __key_cont, __mapped_cont) {}

  template <class _Allocator>
    requires __allocator_ctor_constraint<_Allocator>
  _LIBCPP_HIDE_FROM_ABI
  flat_map(sorted_unique_t,
           const key_container_type& __key_cont,
           const mapped_container_type& __mapped_cont,
           const key_compare& __comp,
           const _Allocator& __alloc)
      : flat_map(__ctor_uses_allocator_tag{}, __alloc, __key_cont, __mapped_cont, __comp) {}

  _LIBCPP_HIDE_FROM_ABI explicit flat_map(const key_compare& __comp) : __containers_(), __compare_(__comp) {}

  template <class _Allocator>
    requires __allocator_ctor_constraint<_Allocator>
  _LIBCPP_HIDE_FROM_ABI flat_map(const key_compare& __comp, const _Allocator& __alloc)
      : flat_map(__ctor_uses_allocator_empty_tag{}, __alloc, __comp) {}

  template <class _Allocator>
    requires __allocator_ctor_constraint<_Allocator>
  _LIBCPP_HIDE_FROM_ABI explicit flat_map(const _Allocator& __alloc)
      : flat_map(__ctor_uses_allocator_empty_tag{}, __alloc) {}

  template <input_iterator _InputIterator>
  _LIBCPP_HIDE_FROM_ABI
  flat_map(_InputIterator __first, _InputIterator __last, const key_compare& __comp = key_compare())
      : __containers_(), __compare_(__comp) {
    insert(__first, __last);
  }

  template <input_iterator _InputIterator, class _Allocator>
    requires __allocator_ctor_constraint<_Allocator>
  _LIBCPP_HIDE_FROM_ABI
  flat_map(_InputIterator __first, _InputIterator __last, const key_compare& __comp, const _Allocator& __alloc)
      : flat_map(__ctor_uses_allocator_empty_tag{}, __alloc, __comp) {
    insert(__first, __last);
  }

  template <input_iterator _InputIterator, class _Allocator>
    requires __allocator_ctor_constraint<_Allocator>
  _LIBCPP_HIDE_FROM_ABI flat_map(_InputIterator __first, _InputIterator __last, const _Allocator& __alloc)
      : flat_map(__ctor_uses_allocator_empty_tag{}, __alloc) {
    insert(__first, __last);
  }

  template <_ContainerCompatibleRange<value_type> _Range>
  _LIBCPP_HIDE_FROM_ABI flat_map(from_range_t __fr, _Range&& __rg)
      : flat_map(__fr, std::forward<_Range>(__rg), key_compare()) {}

  template <_ContainerCompatibleRange<value_type> _Range, class _Allocator>
    requires __allocator_ctor_constraint<_Allocator>
  _LIBCPP_HIDE_FROM_ABI flat_map(from_range_t, _Range&& __rg, const _Allocator& __alloc)
      : flat_map(__ctor_uses_allocator_empty_tag{}, __alloc) {
    insert_range(std::forward<_Range>(__rg));
  }

  template <_ContainerCompatibleRange<value_type> _Range>
  _LIBCPP_HIDE_FROM_ABI flat_map(from_range_t, _Range&& __rg, const key_compare& __comp) : flat_map(__comp) {
    insert_range(std::forward<_Range>(__rg));
  }

  template <_ContainerCompatibleRange<value_type> _Range, class _Allocator>
  _LIBCPP_HIDE_FROM_ABI flat_map(from_range_t, _Range&& __rg, const key_compare& __comp, const _Allocator& __alloc)
      : flat_map(__ctor_uses_allocator_empty_tag{}, __alloc, __comp) {
    insert_range(std::forward<_Range>(__rg));
  }

  template <input_iterator _InputIterator>
  _LIBCPP_HIDE_FROM_ABI flat_map(
      sorted_unique_t __s, _InputIterator __first, _InputIterator __last, const key_compare& __comp = key_compare())
      : __containers_(), __compare_(__comp) {
    insert(__s, __first, __last);
  }
  template <input_iterator _InputIterator, class _Allocator>
    requires __allocator_ctor_constraint<_Allocator>
  _LIBCPP_HIDE_FROM_ABI
  flat_map(sorted_unique_t __s,
           _InputIterator __first,
           _InputIterator __last,
           const key_compare& __comp,
           const _Allocator& __alloc)
      : flat_map(__ctor_uses_allocator_empty_tag{}, __alloc, __comp) {
    insert(__s, __first, __last);
  }

  template <input_iterator _InputIterator, class _Allocator>
    requires __allocator_ctor_constraint<_Allocator>
  _LIBCPP_HIDE_FROM_ABI
  flat_map(sorted_unique_t __s, _InputIterator __first, _InputIterator __last, const _Allocator& __alloc)
      : flat_map(__ctor_uses_allocator_empty_tag{}, __alloc) {
    insert(__s, __first, __last);
  }

  _LIBCPP_HIDE_FROM_ABI flat_map(initializer_list<value_type> __il, const key_compare& __comp = key_compare())
      : flat_map(__il.begin(), __il.end(), __comp) {}

  template <class _Allocator>
    requires __allocator_ctor_constraint<_Allocator>
  _LIBCPP_HIDE_FROM_ABI
  flat_map(initializer_list<value_type> __il, const key_compare& __comp, const _Allocator& __alloc)
      : flat_map(__il.begin(), __il.end(), __comp, __alloc) {}

  template <class _Allocator>
    requires __allocator_ctor_constraint<_Allocator>
  _LIBCPP_HIDE_FROM_ABI flat_map(initializer_list<value_type> __il, const _Allocator& __alloc)
      : flat_map(__il.begin(), __il.end(), __alloc) {}

  _LIBCPP_HIDE_FROM_ABI
  flat_map(sorted_unique_t __s, initializer_list<value_type> __il, const key_compare& __comp = key_compare())
      : flat_map(__s, __il.begin(), __il.end(), __comp) {}

  template <class _Allocator>
    requires __allocator_ctor_constraint<_Allocator>
  _LIBCPP_HIDE_FROM_ABI
  flat_map(sorted_unique_t __s, initializer_list<value_type> __il, const key_compare& __comp, const _Allocator& __alloc)
      : flat_map(__s, __il.begin(), __il.end(), __comp, __alloc) {}

  template <class _Allocator>
    requires __allocator_ctor_constraint<_Allocator>
  _LIBCPP_HIDE_FROM_ABI flat_map(sorted_unique_t __s, initializer_list<value_type> __il, const _Allocator& __alloc)
      : flat_map(__s, __il.begin(), __il.end(), __alloc) {}

  _LIBCPP_HIDE_FROM_ABI flat_map& operator=(initializer_list<value_type> __il) {
    clear();
    insert(__il);
    return *this;
  }

  // iterators
  _LIBCPP_HIDE_FROM_ABI iterator begin() noexcept {
    return iterator(__containers_.keys.begin(), __containers_.values.begin());
  }

  _LIBCPP_HIDE_FROM_ABI const_iterator begin() const noexcept {
    return const_iterator(__containers_.keys.begin(), __containers_.values.begin());
  }

  _LIBCPP_HIDE_FROM_ABI iterator end() noexcept {
    return iterator(__containers_.keys.end(), __containers_.values.end());
  }

  _LIBCPP_HIDE_FROM_ABI const_iterator end() const noexcept {
    return const_iterator(__containers_.keys.end(), __containers_.values.end());
  }

  _LIBCPP_HIDE_FROM_ABI reverse_iterator rbegin() noexcept { return reverse_iterator(end()); }
  _LIBCPP_HIDE_FROM_ABI const_reverse_iterator rbegin() const noexcept { const_reverse_iterator(end()); }
  _LIBCPP_HIDE_FROM_ABI reverse_iterator rend() noexcept { return reverse_iterator(begin()); }
  _LIBCPP_HIDE_FROM_ABI const_reverse_iterator rend() const noexcept { return const_reverse_iterator(begin()); }

  _LIBCPP_HIDE_FROM_ABI const_iterator cbegin() const noexcept { return begin(); }
  _LIBCPP_HIDE_FROM_ABI const_iterator cend() const noexcept { return end(); }
  _LIBCPP_HIDE_FROM_ABI const_reverse_iterator crbegin() const noexcept { return const_reverse_iterator(end()); }
  _LIBCPP_HIDE_FROM_ABI const_reverse_iterator crend() const noexcept { return const_reverse_iterator(begin()); }

  // [flat.map.capacity], capacity
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI bool empty() const noexcept { return __containers_.keys.empty(); }

  _LIBCPP_HIDE_FROM_ABI size_type size() const noexcept { return __containers_.keys.size(); }

  _LIBCPP_HIDE_FROM_ABI size_type max_size() const noexcept {
    return std::min<size_type>(__containers_.keys.max_size(), __containers_.values.max_size());
  }

  // [flat.map.access], element access
  _LIBCPP_HIDE_FROM_ABI mapped_type& operator[](const key_type& __x) { return try_emplace(__x).first->second; }

  _LIBCPP_HIDE_FROM_ABI mapped_type& operator[](key_type&& __x) { return try_emplace(std::move(__x)).first->second; }

  template <class _Kp>
    requires __is_compare_transparent
  _LIBCPP_HIDE_FROM_ABI mapped_type& operator[](_Kp&& __x) {
    return try_emplace(std::forward<_Kp>(__x)).first->second;
  }

  _LIBCPP_HIDE_FROM_ABI mapped_type& at(const key_type& __x) {
    auto __it = find(__x);
    if (__it == end()) {
      __throw_out_of_range("flat_map::at(const key_type&): Key does not exist");
    }
    return (*__it).second;
  }

  _LIBCPP_HIDE_FROM_ABI const mapped_type& at(const key_type& __x) const {
    auto __it = find(__x);
    if (__it == end()) {
      __throw_out_of_range("flat_map::at(const key_type&) const: Key does not exist");
    }
    return (*__it).second;
  }

  template <class _Kp>
    requires __is_compare_transparent
  _LIBCPP_HIDE_FROM_ABI mapped_type& at(const _Kp& __x) {
    static_assert(requires { find(__x); }, "flat_map::at(const K& x): find(x) needs to be well-formed");
    auto __it = find(__x);
    if (__it == end()) {
      __throw_out_of_range("flat_map::at(const K&): Key does not exist");
    }
    return (*__it).second;
  }

  template <class _Kp>
    requires __is_compare_transparent
  _LIBCPP_HIDE_FROM_ABI const mapped_type& at(const _Kp& __x) const {
    static_assert(requires { find(__x); }, "flat_map::at(const K& x) const: find(x) needs to be well-formed");
    auto __it = find(__x);
    if (__it == end()) {
      __throw_out_of_range("flat_map::at(const K&) const: Key does not exist");
    }
    return (*__it).second;
  }

  // [flat.map.modifiers], modifiers
  template <class... _Args>
    requires is_constructible_v<pair<key_type, mapped_type>, _Args...>
  _LIBCPP_HIDE_FROM_ABI pair<iterator, bool> emplace(_Args&&... __args) {
    std::pair<key_type, mapped_type> __pair(std::forward<_Args>(__args)...);
    return __binary_search_emplace_impl(std::move(__pair));
  }

  template <class... _Args>
    requires is_constructible_v<pair<key_type, mapped_type>, _Args...>
  _LIBCPP_HIDE_FROM_ABI iterator emplace_hint(const_iterator __hint, _Args&&... __args) {
    std::pair<key_type, mapped_type> __pair(std::forward<_Args>(__args)...);
    if (__is_hint_correct(__hint, __pair.first)) {
      if (__compare_(__pair.first, __hint->first)) {
        return __emplace_impl(__hint, std::move(__pair));
      } else {
        // key equals
        auto __dist = __hint - cbegin();
        return iterator(__containers_.keys.begin() + __dist, __containers_.values.begin() + __dist);
      }
    } else {
      return __binary_search_emplace_impl(std::move(__pair)).first;
    }
  }

  _LIBCPP_HIDE_FROM_ABI pair<iterator, bool> insert(const value_type& __x) { return emplace(__x); }

  _LIBCPP_HIDE_FROM_ABI pair<iterator, bool> insert(value_type&& __x) { return emplace(std::move(__x)); }

  _LIBCPP_HIDE_FROM_ABI iterator insert(const_iterator __hint, const value_type& __x) {
    return emplace_hint(__hint, __x);
  }

  _LIBCPP_HIDE_FROM_ABI iterator insert(const_iterator __hint, value_type&& __x) {
    return emplace_hint(__hint, std::move(__x));
  }

  template <class _Pp>
    requires is_constructible_v<pair<key_type, mapped_type>, _Pp>
  _LIBCPP_HIDE_FROM_ABI pair<iterator, bool> insert(_Pp&& __x) {
    return emplace(std::forward<_Pp>(__x));
  }

  template <class _Pp>
    requires is_constructible_v<pair<key_type, mapped_type>, _Pp>
  _LIBCPP_HIDE_FROM_ABI iterator insert(const_iterator __hint, _Pp&& __x) {
    return emplace_hint(__hint, std::forward<_Pp>(__x));
  }

  template <input_iterator _InputIterator>
  _LIBCPP_HIDE_FROM_ABI void insert(_InputIterator __first, _InputIterator __last) {
    if constexpr (sized_sentinel_for<_InputIterator, _InputIterator>) {
      __reserve_impl(__last - __first);
    }

    for (; __first != __last; ++__first) {
      __binary_search_emplace_impl(value_type(*__first));
    }
  }

  template <input_iterator _InputIterator>
  void insert(sorted_unique_t, _InputIterator __first, _InputIterator __last) {
    if constexpr (sized_sentinel_for<_InputIterator, _InputIterator>) {
      __reserve_impl(__last - __first);
    }

    auto __it = begin();
    while (__first != __last) {
      value_type __pair(*__first);
      auto __end = end();
      __it       = ranges::lower_bound(__it, __end, __pair.first, __compare_, [](const auto& __p) -> decltype(auto) {
        return std::get<0>(__p);
      });
      if (__it == __end || __compare_(__pair.first, __it->first)) {
        __it = __emplace_impl(__it, std::move(__pair));
      }
      ++__it;
      ++__first;
    }
  }

  template <_ContainerCompatibleRange<value_type> _Range>
  _LIBCPP_HIDE_FROM_ABI void insert_range(_Range&& __range) {
    if constexpr (ranges::sized_range<_Range>) {
      __reserve_impl(ranges::size(__range));
    }

    auto __last = ranges::end(__range);
    for (auto __it = ranges::begin(__range); __it != __last; ++__it) {
      __binary_search_emplace_impl(value_type(*__it));
    }
  }

  _LIBCPP_HIDE_FROM_ABI void insert(initializer_list<value_type> __il) { insert(__il.begin(), __il.end()); }

  _LIBCPP_HIDE_FROM_ABI void insert(sorted_unique_t __s, initializer_list<value_type> __il) {
    insert(__s, __il.begin(), __il.end());
  }

  _LIBCPP_HIDE_FROM_ABI containers extract() && {
#  ifndef _LIBCPP_HAS_NO_EXCEPTIONS
    try {
#  endif // _LIBCPP_HAS_NO_EXCEPTIONS
      return std::move(__containers_);
#  ifndef _LIBCPP_HAS_NO_EXCEPTIONS
    } catch (...) {
      clear();
      throw;
    }
#  endif // _LIBCPP_HAS_NO_EXCEPTIONS
  }

  _LIBCPP_HIDE_FROM_ABI void replace(key_container_type&& __key_cont, mapped_container_type&& __mapped_cont) {
#  ifndef _LIBCPP_HAS_NO_EXCEPTIONS
    try {
#  endif // _LIBCPP_HAS_NO_EXCEPTIONS
      __containers_.keys   = std::move(__key_cont);
      __containers_.values = std::move(__mapped_cont);
#  ifndef _LIBCPP_HAS_NO_EXCEPTIONS
    } catch (...) {
      clear();
      throw;
    }
#  endif // _LIBCPP_HAS_NO_EXCEPTIONS
  }

  template <class... _Args>
    requires is_constructible_v<mapped_type, _Args...>
  _LIBCPP_HIDE_FROM_ABI pair<iterator, bool> try_emplace(const key_type& __key, _Args&&... __args) {
    return __binary_search_try_emplace_impl(__key, std::forward<_Args>(__args)...);
  }

  template <class... _Args>
    requires is_constructible_v<mapped_type, _Args...>
  _LIBCPP_HIDE_FROM_ABI pair<iterator, bool> try_emplace(key_type&& __key, _Args&&... __args) {
    return __binary_search_try_emplace_impl(std::move(__key), std::forward<_Args>(__args)...);
  }

  template <class _Kp, class... _Args>
    requires __is_compare_transparent && is_constructible_v<key_type, _Kp> &&
             is_constructible_v<mapped_type, _Args...> && is_convertible_v<_Kp&&, const_iterator> &&
             is_convertible_v<_Kp&&, iterator>
  _LIBCPP_HIDE_FROM_ABI pair<iterator, bool> try_emplace(_Kp&& __key, _Args&&... __args) {
    return __binary_search_try_emplace_impl(std::forward<_Kp>(__key), std::forward<_Args>(__args)...);
  }

  template <class... _Args>
    requires is_constructible_v<mapped_type, _Args...>
  _LIBCPP_HIDE_FROM_ABI iterator try_emplace(const_iterator __hint, const key_type& __key, _Args&&... __args) {
    return try_emplace_hint_impl(__hint, __key, std::forward<_Args>(__args)...);
  }

  template <class... _Args>
    requires is_constructible_v<mapped_type, _Args...>
  _LIBCPP_HIDE_FROM_ABI iterator try_emplace(const_iterator __hint, key_type&& __key, _Args&&... __args) {
    return try_emplace_hint_impl(__hint, std::move(__key), std::forward<_Args>(__args)...);
  }

  template <class _Kp, class... _Args>
    requires __is_compare_transparent && is_constructible_v<key_type, _Kp> && is_constructible_v<mapped_type, _Args...>
  _LIBCPP_HIDE_FROM_ABI iterator try_emplace(const_iterator __hint, _Kp&& __key, _Args&&... __args) {
    return try_emplace_hint_impl(__hint, std::forward<_Kp>(__key), std::forward<_Args>(__args)...);
  }

  template <class _Mapped>
    requires is_assignable_v<mapped_type&, _Mapped> && is_constructible_v<mapped_type, _Mapped>
  _LIBCPP_HIDE_FROM_ABI pair<iterator, bool> insert_or_assign(const key_type& __key, _Mapped&& __obj) {
    return __insert_or_assign_impl(__key, std::forward<_Mapped>(__obj));
  }

  template <class _Mapped>
    requires is_assignable_v<mapped_type&, _Mapped> && is_constructible_v<mapped_type, _Mapped>
  _LIBCPP_HIDE_FROM_ABI pair<iterator, bool> insert_or_assign(key_type&& __key, _Mapped&& __obj) {
    return __insert_or_assign_impl(std::move(__key), std::forward<_Mapped>(__obj));
  }

  template <class _Kp, class _Mapped>
    requires __is_compare_transparent && is_constructible_v<key_type, _Kp> && is_assignable_v<mapped_type&, _Mapped> &&
             is_constructible_v<mapped_type, _Mapped>
  _LIBCPP_HIDE_FROM_ABI pair<iterator, bool> insert_or_assign(_Kp&& __key, _Mapped&& __obj) {
    return __insert_or_assign_impl(std::forward<_Kp>(__key), std::forward<_Mapped>(__obj));
  }

  template <class _Mapped>
    requires is_assignable_v<mapped_type&, _Mapped> && is_constructible_v<mapped_type, _Mapped>
  _LIBCPP_HIDE_FROM_ABI iterator insert_or_assign(const_iterator __hint, const key_type& __key, _Mapped&& __obj) {
    return __insert_or_assign_impl(__key, std::forward<_Mapped>(__obj), __hint).first;
  }

  template <class _Mapped>
    requires is_assignable_v<mapped_type&, _Mapped> && is_constructible_v<mapped_type, _Mapped>
  _LIBCPP_HIDE_FROM_ABI iterator insert_or_assign(const_iterator __hint, key_type&& __key, _Mapped&& __obj) {
    return __insert_or_assign_impl(std::move(__key), std::forward<_Mapped>(__obj), __hint).first;
  }

  template <class _Kp, class _Mapped>
    requires __is_compare_transparent && is_constructible_v<key_type, _Kp> && is_assignable_v<mapped_type&, _Mapped> &&
             is_constructible_v<mapped_type, _Mapped>
  _LIBCPP_HIDE_FROM_ABI iterator insert_or_assign(const_iterator __hint, _Kp&& __key, _Mapped&& __obj) {
    return __insert_or_assign_impl(std::forward<_Kp>(__key), std::forward<_Mapped>(__obj), __hint).first;
  }

  _LIBCPP_HIDE_FROM_ABI iterator erase(iterator __position) {
    return __erase_impl(__position.__key_iter_, __position.__mapped_iter);
  }

  _LIBCPP_HIDE_FROM_ABI iterator erase(const_iterator __position) {
    return __erase_impl(__position.__key_iter_, __position.__mapped_iter);
  }

  _LIBCPP_HIDE_FROM_ABI size_type erase(const key_type& __x) {
    auto __iter = find(__x);
    if (__iter != end()) {
      erase(__iter);
      return 1;
    }
    return 0;
  }

  template <class _Kp>
    requires __is_compare_transparent
  _LIBCPP_HIDE_FROM_ABI size_type erase(_Kp&& __x) {
    auto __iter = find(__x);
    if (__iter != end()) {
      erase(__iter);
      return 1;
    }
    return 0;
  }

  _LIBCPP_HIDE_FROM_ABI iterator erase(const_iterator __first, const_iterator __last) {
#  ifndef _LIBCPP_HAS_NO_EXCEPTIONS
    try {
#  endif // _LIBCPP_HAS_NO_EXCEPTIONS
      auto __key_it    = __containers_.keys.erase(__first.__key_iter, __last.__key_iter);
      auto __mapped_it = __containers_.values.erase(__first.__mapped_iter, __last.__mapped_iter);
      return iterator(std::move(__key_it), std::move(__mapped_it));
#  ifndef _LIBCPP_HAS_NO_EXCEPTIONS
    } catch (const exception& __ex) {
      clear();
      throw flat_map_restore_error(
          std::string("flat_map::erase: "
                      "Unable to restore flat_map to previous state. Clear out the containers to make the two "
                      "containers consistent. Reason: ") +
          __ex.what());
    } catch (...) {
      clear();
      throw flat_map_restore_error(
          "flat_map::erase: "
          "Unable to restore flat_map to previous state. Clear out the containers to make the two "
          "containers consistent.");
    }
#  endif // _LIBCPP_HAS_NO_EXCEPTIONS
  }

  _LIBCPP_HIDE_FROM_ABI void swap(flat_map& __y) noexcept {
#  ifndef _LIBCPP_HAS_NO_EXCEPTIONS
    try {
#  endif // _LIBCPP_HAS_NO_EXCEPTIONS
      using std::swap;
      swap(__compare_, __y.__compare_);
      swap(__containers_.keys, __y.__containers_.keys);
      swap(__containers_.values, __y.__containers_.values);
#  ifndef _LIBCPP_HAS_NO_EXCEPTIONS
    } catch (...) {
      clear();
      __y.clear();
    }
#  endif // _LIBCPP_HAS_NO_EXCEPTIONS
  }

  _LIBCPP_HIDE_FROM_ABI void clear() noexcept {
    __containers_.keys.clear();
    __containers_.values.clear();
  }

  // observers
  _LIBCPP_HIDE_FROM_ABI key_compare key_comp() const { return __compare_; }
  _LIBCPP_HIDE_FROM_ABI value_compare value_comp() const { return value_compare(__compare_); }
  _LIBCPP_HIDE_FROM_ABI const key_container_type& keys() const noexcept { return __containers_.keys; }
  _LIBCPP_HIDE_FROM_ABI const mapped_container_type& values() const noexcept { return __containers_.values; }

  // map operations
  _LIBCPP_HIDE_FROM_ABI iterator find(const key_type& __x) { return __find_impl(*this, __x); }

  _LIBCPP_HIDE_FROM_ABI const_iterator find(const key_type& __x) const { return __find_impl(*this, __x); }

  template <class _Kp>
    requires __is_compare_transparent
  _LIBCPP_HIDE_FROM_ABI iterator find(const _Kp& __x) {
    return __find_impl(*this, __x);
  }

  template <class _Kp>
    requires __is_compare_transparent
  _LIBCPP_HIDE_FROM_ABI const_iterator find(const _Kp& __x) const {
    return __find_impl(*this, __x);
  }

  _LIBCPP_HIDE_FROM_ABI size_type count(const key_type& __x) const { return contains(__x) ? 1 : 0; }

  template <class _Kp>
  _LIBCPP_HIDE_FROM_ABI size_type count(const _Kp& __x) const {
    return contains(__x) ? 1 : 0;
  }

  _LIBCPP_HIDE_FROM_ABI bool contains(const key_type& __x) const { return find(__x) != end(); }

  template <class _Kp>
  _LIBCPP_HIDE_FROM_ABI bool contains(const _Kp& __x) const {
    return find(__x) != end();
  }

  _LIBCPP_HIDE_FROM_ABI iterator lower_bound(const key_type& __x) { return __lower_bound_impl<iterator>(*this, __x); }

  _LIBCPP_HIDE_FROM_ABI const_iterator lower_bound(const key_type& __x) const {
    return __lower_bound_impl<const_iterator>(*this, __x);
  }

  template <class _Kp>
    requires __is_compare_transparent
  _LIBCPP_HIDE_FROM_ABI iterator lower_bound(const _Kp& __x) {
    return __lower_bound_impl<iterator>(*this, __x);
  }

  template <class _Kp>
    requires __is_compare_transparent
  _LIBCPP_HIDE_FROM_ABI const_iterator lower_bound(const _Kp& __x) const {
    return __lower_bound_impl<const_iterator>(*this, __x);
  }

  _LIBCPP_HIDE_FROM_ABI iterator upper_bound(const key_type& __x) { return __upper_bound_impl<iterator>(*this, __x); }

  _LIBCPP_HIDE_FROM_ABI const_iterator upper_bound(const key_type& __x) const {
    return __upper_bound_impl<const_iterator>(*this, __x);
  }

  template <class _Kp>
    requires __is_compare_transparent
  _LIBCPP_HIDE_FROM_ABI iterator upper_bound(const _Kp& __x) {
    return __upper_bound_impl<iterator>(*this, __x);
  }

  template <class _Kp>
    requires __is_compare_transparent
  _LIBCPP_HIDE_FROM_ABI const_iterator upper_bound(const _Kp& __x) const {
    return __upper_bound_impl<iterator>(*this, __x);
  }

  _LIBCPP_HIDE_FROM_ABI pair<iterator, iterator> equal_range(const key_type& __x) {
    return __equal_range_impl(*this, __x);
  }

  _LIBCPP_HIDE_FROM_ABI pair<const_iterator, const_iterator> equal_range(const key_type& __x) const {
    return __equal_range_impl(*this, __x);
  }

  template <class _Kp>
    requires __is_compare_transparent
  _LIBCPP_HIDE_FROM_ABI pair<iterator, iterator> equal_range(const _Kp& __x) {
    return __equal_range_impl(*this, __x);
  }
  template <class _Kp>
    requires __is_compare_transparent
  _LIBCPP_HIDE_FROM_ABI pair<const_iterator, const_iterator> equal_range(const _Kp& __x) const {
    return __equal_range_impl(*this, __x);
  }

  friend _LIBCPP_HIDE_FROM_ABI bool operator==(const flat_map& __x, const flat_map& __y) {
    return ranges::equal(__x, __y);
  }

  friend _LIBCPP_HIDE_FROM_ABI __synth_three_way_result<value_type>
  operator<=>(const flat_map& __x, const flat_map& __y) {
    return ranges::lexicographical_compare(__x, __y);
  }

  friend _LIBCPP_HIDE_FROM_ABI void swap(flat_map& __x, flat_map& __y) noexcept { __x.swap(__y); }

private:
  struct __ctor_uses_allocator_tag {
    explicit __ctor_uses_allocator_tag() = default;
  };
  struct __ctor_uses_allocator_empty_tag {
    explicit __ctor_uses_allocator_empty_tag() = default;
  };
  _LIBCPP_HIDE_FROM_ABI void __sort_and_unique() {
    auto __zv = ranges::views::zip(__containers_.keys, __containers_.values);
    ranges::stable_sort(__zv, __compare_, [](const auto& __p) -> decltype(auto) { return std::get<0>(__p); });
    auto __it   = ranges::unique(__zv, __key_equiv(__compare_)).begin();
    auto __dist = ranges::distance(__zv.begin(), __it);
    __containers_.keys.erase(__containers_.keys.begin() + __dist, __containers_.keys.end());
    __containers_.values.erase(__containers_.values.begin() + __dist, __containers_.values.end());
  }

  template <class _Allocator, class _KeyCont, class _MappedCont, class... _CompArg>
  _LIBCPP_HIDE_FROM_ABI
  flat_map(__ctor_uses_allocator_tag,
           const _Allocator& __alloc,
           _KeyCont&& __key_cont,
           _MappedCont&& __mapped_cont,
           _CompArg&&... __comp)
      : __containers_{.keys = std::make_obj_using_allocator<key_container_type>(
                          __alloc, std::forward<_KeyCont>(__key_cont)),
                      .values = std::make_obj_using_allocator<mapped_container_type>(
                          __alloc, std::forward<_MappedCont>(__mapped_cont))},
        __compare_(std::forward<_CompArg>(__comp)...) {}

  template <class _Allocator, class... _CompArg>
  _LIBCPP_HIDE_FROM_ABI flat_map(__ctor_uses_allocator_empty_tag, const _Allocator& __alloc, _CompArg&&... __comp)
      : __containers_{.keys   = std::make_obj_using_allocator<key_container_type>(__alloc),
                      .values = std::make_obj_using_allocator<mapped_container_type>(__alloc)},
        __compare_(std::forward<_CompArg>(__comp)...) {}

  template <class _Self, class _Kp>
  _LIBCPP_HIDE_FROM_ABI static auto __find_impl(_Self&& __self, const _Kp& __key) {
    auto __it   = __self.lower_bound(__key);
    auto __last = __self.end();
    if (__it == __last || __self.__compare_(__key, __it->first)) {
      return __last;
    }
    return __it;
  }

  template <class _Self, class _Kp>
  _LIBCPP_HIDE_FROM_ABI static auto __equal_range_impl(_Self&& __self, const _Kp& __key) {
    auto __it   = __self.lower_bound(__key);
    auto __last = __self.end();
    if (__it == __last || __self.__compare_(__key, __it->first)) {
      return std::make_pair(std::move(__it), std::move(__last));
    }
    return std::make_pair(__it, std::next(__it));
  }

  template <class _Res, class _Self, class _Kp>
  _LIBCPP_HIDE_FROM_ABI static _Res __lower_bound_impl(_Self&& __self, _Kp& __x) {
    return __binary_search_impl<_Res>(ranges::lower_bound, __self, __x);
  }

  template <class _Res, class _Self, class _Kp>
  _LIBCPP_HIDE_FROM_ABI static _Res __upper_bound_impl(_Self&& __self, _Kp& __x) {
    return __binary_search_impl<_Res>(ranges::upper_bound, __self, __x);
  }

  template <class _Kp>
  _LIBCPP_HIDE_FROM_ABI bool __is_hint_correct(const_iterator __hint, _Kp&& __key) {
    if (__hint != cbegin() && !__compare_(std::prev(__hint)->first, __key)) {
      return false;
    }
    if (__hint != cend() && __compare(__hint->first, __key)) {
      return false;
    }
    return true;
  }

  template <class _Res, class _Fn, class _Self, class _Kp>
  _LIBCPP_HIDE_FROM_ABI static _Res __binary_search_impl(_Fn __search_fn, _Self&& __self, _Kp& __x) {
    auto __key_iter = __search_fn(__self.__containers_.keys, __x, __self.__compare_);
    auto __mapped_iter =
        __self.__containers_.values.begin() +
        static_cast<ranges::range_difference_t<mapped_container_type>>(
            ranges::distance(__self.__containers_.keys.begin(), __key_iter));

    return _Res(std::move(__key_iter), std::move(__mapped_iter));
  }

  _LIBCPP_HIDE_FROM_ABI pair<iterator, bool> __binary_search_emplace_impl(std::pair<key_type, mapped_type>&& __pair) {
    if (auto __it = lower_bound(__pair.first); __it == end() || __compare_(__pair.first, (*__it).first)) {
      return pair<iterator, bool>(__emplace_impl(__it, std::move(__pair)), true);
    } else {
      return pair<iterator, bool>(std::move(__it), false);
    }
  }

  template <class _Iter>
  _LIBCPP_HIDE_FROM_ABI iterator __emplace_impl(_Iter&& __it, std::pair<key_type, mapped_type>&& __pair) {
    return __try_emplace_impl(__it.__key_iter_, __it.__mapped_iter_, std::move(__pair.first), std::move(__pair.second));
  }

  template <class _KeyArg, class... _MArgs>
  _LIBCPP_HIDE_FROM_ABI pair<iterator, bool>
  __binary_search_try_emplace_impl(_KeyArg&& __key, _MArgs&&... __mapped_args) {
    auto __key_it    = ranges::lower_bound(__containers_.keys, __key, __compare_);
    auto __mapped_it = __containers_.values.begin() + ranges::distance(__containers_.keys.begin(), __key_it);

    if (__key_it == __containers_.keys.end() || __compare_(__key, *__key_it)) {
      return pair<iterator, bool>(
          __try_emplace_impl(std::move(__key_it),
                             std::move(__mapped_it),
                             std::forward<_KeyArg>(__key),
                             std::forward<_MArgs>(__mapped_args)...),
          true);
    } else {
      return pair<iterator, bool>(iterator(std::move(__key_it), std::move(__mapped_it)), false);
    }
  }

  template <class _Kp, class... _Args>
  _LIBCPP_HIDE_FROM_ABI iterator try_emplace_hint_impl(const_iterator __hint, _Kp&& __key, _Args&&... __args) {
    if (__is_hint_correct(__hint, __key)) {
      if (__compare_(__key, __hint->first)) {
        return __try_emplace_impl(
            __hint.__key_iter_, __hint.__mapped_iter_, std::forward<_Kp>(__key), std::forward<_Args>(__args)...);
      } else {
        // key equals
        auto __dist = __hint - cbegin();
        return iterator(__containers_.keys.begin() + __dist, __containers_.values.begin() + __dist);
      }
    } else {
      __binary_search_try_emplace_impl(std::forward<_Kp>(__key), std::forward<_Args>(__args)...).first;
    }
  }

  template <class _Container>
  static consteval bool __failed_emplacement_has_side_effects() {
    // [container.reqmts] If an exception is thrown by an insert() or emplace() function while inserting a single
    // element, that function has no effects. Except that there is exceptional cases...

    // according to http://eel.is/c++draft/deque.modifiers#3 and http://eel.is/c++draft/vector.modifiers#2,
    // the only exceptions that can cause side effects on single emplacement are by move constructors of
    // non-Cpp17CopyInsertable T

    using _Element = typename _Container::value_type;
    if constexpr (is_nothrow_move_constructible_v<_Element>) {
      return false;
    } else {
      if constexpr (requires { typename _Container::allocator_type; }) {
        return !__is_cpp17_copy_insertable<typename _Container::allocator_type>::value;
      } else {
        return !__is_cpp17_copy_insertable<std::allocator<_Element>>::value;
      }
    }
  }

  struct flat_map_restore_error : runtime_error {
    using runtime_error::runtime_error;
  };

  template <class _Container, class _Iter, class... _Args>
  _LIBCPP_HIDE_FROM_ABI ranges::iterator_t<_Container>
  __safe_emplace(_Container& __container, _Iter&& __iter, _Args&&... __args) {
    if constexpr (!__failed_emplacement_has_side_effects<_Container>()) {
      // just let the exception be thrown as the container is still in its original state on exception
      return __container.emplace(__iter, std::forward<_Args>(__args)...);
    } else {
#  ifndef _LIBCPP_HAS_NO_EXCEPTIONS
      try {
#  endif // _LIBCPP_HAS_NO_EXCEPTIONS
#  ifndef _LIBCPP_HAS_NO_EXCEPTIONS
      } catch (const exception& __ex) {
        // The container might be in some unknown state and we can't get flat_map into consistent state
        // because we have two containers. The only possible solution is to clear them out
        clear();
        throw flat_map_restore_error(
            std::string("flat_map::emplace: Emplacement on the underlying container has failed and has side effect. "
                        "Unable to restore flat_map to previous state. Clear out the containers to make the two "
                        "containers consistent. Reason: ") +
            __ex.what());
      } catch (...) {
        // The container might be in some unknown state and we can't get flat_map into consistent state
        // because we have two containers. The only possible solution is to clear them out
        clear();
        throw flat_map_restore_error(
            "flat_map::emplace: Emplacement on the underlying container has failed and has side effect. "
            "Unable to restore flat_map to previous state. Clear out the containers to make the two "
            "containers consistent.");
      }
#  endif // _LIBCPP_HAS_NO_EXCEPTIONS
    }
  }

  template <class _Container, class _Iter>
  _LIBCPP_HIDE_FROM_ABI auto __safe_erase(_Container& __container, _Iter&& __iter) {
    // [container.reqmts] No erase(), clear(), pop_back() or pop_front() function throws an exception,
    // except that there are exceptional cases

    // http://eel.is/c++draft/deque.modifiers#5
    // http://eel.is/c++draft/vector.modifiers#4

#  ifndef _LIBCPP_HAS_NO_EXCEPTIONS
    try {
#  endif // _LIBCPP_HAS_NO_EXCEPTIONS
      return __container.erase(__iter);
#  ifndef _LIBCPP_HAS_NO_EXCEPTIONS
    } catch (const exception& __ex) {
      // The container might be in some unknown state and we can't get flat_map into consistent state
      // because we have two containers. The only possible solution is to clear them out
      clear();
      throw flat_map_restore_error(
          std::string("flat_map: Erasing on the underlying container has failed. "
                      "Unable to restore flat_map to previous state. Clear out the containers to make the two "
                      "containers consistent. Reason: ") +
          __ex.what());
    } catch (...) {
      // The container might be in some unknown state and we can't get flat_map into consistent state
      // because we have two containers. The only possible solution is to clear them out
      clear();
      throw flat_map_restore_error(
          "flat_map: Erasing on the underlying container has failed. "
          "Unable to restore flat_map to previous state. Clear out the containers to make the two "
          "containers consistent.");
    }
#  endif // _LIBCPP_HAS_NO_EXCEPTIONS
  }

  template <class _IterK, class _IterM, class _KeyArg, class... _MArgs>
  _LIBCPP_HIDE_FROM_ABI iterator
  __try_emplace_impl(_IterK&& __it_key, _IterM&& __it_mapped, _KeyArg&& __key, _MArgs&&... __mapped_args) {
    auto __key_it = __safe_emplace(__containers_.keys, __it_key, std::forward<_KeyArg>(__key));

#  ifndef _LIBCPP_HAS_NO_EXCEPTIONS
    try {
#  endif // _LIBCPP_HAS_NO_EXCEPTIONS
      auto __mapped_it = __safe_emplace(__containers_.values, __it_mapped, std::forward<_MArgs>(__mapped_args)...);
      return iterator(std::move(__key_it), std::move(__mapped_it));
#  ifndef _LIBCPP_HAS_NO_EXCEPTIONS
    } catch (const flat_map_restore_error&) {
      // both containers already cleared out
      throw;
    } catch (...) {
      // If the second emplace throws and it has no effects on `values`, we need to erase the emplaced key.
      __safe_erase(__containers_.keys, __key_it);
    }
#  endif // _LIBCPP_HAS_NO_EXCEPTIONS
  }

  template <class _Kp, class _Mapped, class... _Hint>
  _LIBCPP_HIDE_FROM_ABI auto __insert_or_assign_impl(_Kp&& __key, _Mapped&& __mapped, _Hint&&... __hint) {
    auto __r = try_emplace(__hint..., std::forward<_Kp>(__key), std::forward<_Mapped>(__mapped));
    if (!__r.second) {
      __r.first->second = std::forward<_Mapped>(__mapped);
    }
    return __r;
  }

  _LIBCPP_HIDE_FROM_ABI void __reserve_impl(size_t __size) {
    if constexpr (requires { __containers_.keys.reserve(__size); }) {
      __containers_.keys.reserve(__size);
    }

    if constexpr (requires { __containers_.values.reserve(__size); }) {
      __containers_.values.reserve(__size);
    }
  }

  template <class _KIter, class _MIter>
  _LIBCPP_HIDE_FROM_ABI iterator __erase_impl(_KIter __k_iter, _MIter __m_iter) {
    auto __key_iter = __safe_erase(__containers_.keys, __k_iter);
#  ifndef _LIBCPP_HAS_NO_EXCEPTIONS
    try {
#  endif // _LIBCPP_HAS_NO_EXCEPTIONS
      auto __mapped_iter = __safe_erase(__containers_.values, __m_iter);
      return iterator(std::move(__key_iter), std::move(__mapped_iter));
#  ifndef _LIBCPP_HAS_NO_EXCEPTIONS
    } catch (const flat_map_restore_error&) {
      // both containers already cleared out
      throw;
    } catch (...) {
      // If the second erase throws, the first erase already happened. The flat_map is inconsistent.
      clear();
      throw flat_map_restore_error(
          "flat_map::erase: Key has been erased but exception thrown on erasing mapped value. To make flat_map in "
          "consistent state, clear out the flat_map");
    }
#  endif // _LIBCPP_HAS_NO_EXCEPTIONS
  }

  containers __containers_;
  [[no_unique_address]] key_compare __compare_;

  struct __key_equiv {
    __key_equiv(key_compare __c) : __comp_(__c) {}
    bool operator()(const_reference __x, const_reference __y) const {
      return !__comp_(__x.first, __y.first) && !__comp_(__y.first, __x.first);
    }
    key_compare __comp_;
  };
};

template <class _KeyContainer, class _MappedContainer, class _Compare = less<typename _KeyContainer::value_type>>
  requires(!__is_allocator<_Compare>::value && !__is_allocator<_KeyContainer>::value &&
           !__is_allocator<_MappedContainer>::value &&
           is_invocable_v<const _Compare&,
                          const typename _KeyContainer::value_type&,
                          const typename _KeyContainer::value_type&>)
flat_map(_KeyContainer, _MappedContainer, _Compare = _Compare())
    -> flat_map<typename _KeyContainer::value_type,
                typename _MappedContainer::value_type,
                _Compare,
                _KeyContainer,
                _MappedContainer>;

template <class _KeyContainer, class _MappedContainer, class _Allocator>
  requires(uses_allocator_v<_KeyContainer, _Allocator> && uses_allocator_v<_MappedContainer, _Allocator> &&
           !__is_allocator<_KeyContainer>::value && !__is_allocator<_MappedContainer>::value)
flat_map(_KeyContainer, _MappedContainer, _Allocator)
    -> flat_map<typename _KeyContainer::value_type,
                typename _MappedContainer::value_type,
                less<typename _KeyContainer::value_type>,
                _KeyContainer,
                _MappedContainer>;

template <class _KeyContainer, class _MappedContainer, class _Compare, class _Allocator>
  requires(!__is_allocator<_Compare>::value && !__is_allocator<_KeyContainer>::value &&
           !__is_allocator<_MappedContainer>::value && uses_allocator_v<_KeyContainer, _Allocator> &&
           uses_allocator_v<_MappedContainer, _Allocator> &&
           is_invocable_v<const _Compare&,
                          const typename _KeyContainer::value_type&,
                          const typename _KeyContainer::value_type&>)
flat_map(_KeyContainer, _MappedContainer, _Compare, _Allocator)
    -> flat_map<typename _KeyContainer::value_type,
                typename _MappedContainer::value_type,
                _Compare,
                _KeyContainer,
                _MappedContainer>;

template <class _KeyContainer, class _MappedContainer, class _Compare = less<typename _KeyContainer::value_type>>
  requires(!__is_allocator<_Compare>::value && !__is_allocator<_KeyContainer>::value &&
           !__is_allocator<_MappedContainer>::value &&
           is_invocable_v<const _Compare&,
                          const typename _KeyContainer::value_type&,
                          const typename _KeyContainer::value_type&>)
flat_map(sorted_unique_t, _KeyContainer, _MappedContainer, _Compare = _Compare())
    -> flat_map<typename _KeyContainer::value_type,
                typename _MappedContainer::value_type,
                _Compare,
                _KeyContainer,
                _MappedContainer>;

template <class _KeyContainer, class _MappedContainer, class _Allocator>
  requires(uses_allocator_v<_KeyContainer, _Allocator> && uses_allocator_v<_MappedContainer, _Allocator> &&
           !__is_allocator<_KeyContainer>::value && !__is_allocator<_MappedContainer>::value)
flat_map(sorted_unique_t, _KeyContainer, _MappedContainer, _Allocator)
    -> flat_map<typename _KeyContainer::value_type,
                typename _MappedContainer::value_type,
                less<typename _KeyContainer::value_type>,
                _KeyContainer,
                _MappedContainer>;

template <class _KeyContainer, class _MappedContainer, class _Compare, class _Allocator>
  requires(!__is_allocator<_Compare>::value && !__is_allocator<_KeyContainer>::value &&
           !__is_allocator<_MappedContainer>::value && uses_allocator_v<_KeyContainer, _Allocator> &&
           uses_allocator_v<_MappedContainer, _Allocator> &&
           is_invocable_v<const _Compare&,
                          const typename _KeyContainer::value_type&,
                          const typename _KeyContainer::value_type&>)
flat_map(sorted_unique_t, _KeyContainer, _MappedContainer, _Compare, _Allocator)
    -> flat_map<typename _KeyContainer::value_type,
                typename _MappedContainer::value_type,
                _Compare,
                _KeyContainer,
                _MappedContainer>;

template <input_iterator _InputIterator, class _Compare = less<__iter_key_type<_InputIterator>>>
  requires(!__is_allocator<_Compare>::value)
flat_map(_InputIterator, _InputIterator, _Compare = _Compare())
    -> flat_map<__iter_key_type<_InputIterator>, __iter_mapped_type<_InputIterator>, _Compare>;

template <input_iterator _InputIterator, class _Compare = less<__iter_key_type<_InputIterator>>>
  requires(!__is_allocator<_Compare>::value)
flat_map(sorted_unique_t, _InputIterator, _InputIterator, _Compare = _Compare())
    -> flat_map<__iter_key_type<_InputIterator>, __iter_mapped_type<_InputIterator>, _Compare>;

template <ranges::input_range _Range,
          class _Compare   = less<__iter_key_type<_Range>>,
          class _Allocator = allocator<byte>>
  requires(!__is_allocator<_Compare>::value && __is_allocator<_Allocator>::value)
flat_map(from_range_t, _Range&&, _Compare = _Compare(), _Allocator = _Allocator())
    -> flat_map<__range_key_type<_Range>,
                __range_mapped_type<_Range>,
                _Compare,
                vector<__range_key_type<_Range>, __alloc_rebind<_Allocator, __range_key_type<_Range>>>,
                vector<__range_mapped_type<_Range>, __alloc_rebind<_Allocator, __range_mapped_type<_Range>>>>;

template <ranges::input_range _Range, class _Allocator>
  requires __is_allocator<_Allocator>::value
flat_map(from_range_t, _Range&&, _Allocator)
    -> flat_map<__range_key_type<_Range>,
                __range_mapped_type<_Range>,
                less<__range_key_type<_Range>>,
                vector<__range_key_type<_Range>, __alloc_rebind<_Allocator, __range_key_type<_Range>>>,
                vector<__range_mapped_type<_Range>, __alloc_rebind<_Allocator, __range_mapped_type<_Range>>>>;

template <class _Key, class _Tp, class _Compare = less<_Key>>
  requires(!__is_allocator<_Compare>::value)
flat_map(initializer_list<pair<_Key, _Tp>>, _Compare = _Compare()) -> flat_map<_Key, _Tp, _Compare>;

template <class _Key, class _Tp, class _Compare = less<_Key>>
  requires(!__is_allocator<_Compare>::value)
flat_map(sorted_unique_t, initializer_list<pair<_Key, _Tp>>, _Compare = _Compare()) -> flat_map<_Key, _Tp, _Compare>;

template <class _Key, class _Tp, class _Compare, class _KeyContainer, class _MappedContainer, class _Allocator>
struct uses_allocator<flat_map<_Key, _Tp, _Compare, _KeyContainer, _MappedContainer>, _Allocator>
    : bool_constant<uses_allocator_v<_KeyContainer, _Allocator> && uses_allocator_v<_MappedContainer, _Allocator>> {};

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 23

#endif // _LIBCPP___FLAT_MAP_FLAT_MAP_H
