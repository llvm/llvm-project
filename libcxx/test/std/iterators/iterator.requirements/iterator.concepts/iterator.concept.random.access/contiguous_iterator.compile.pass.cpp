//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts

// template<class T>
// concept contiguous_iterator;

#include <iterator>
#include <compare>

#include "test_iterators.h"

static_assert(!std::contiguous_iterator<cpp17_input_iterator<int*>>);
static_assert(!std::contiguous_iterator<cpp20_input_iterator<int*>>);
static_assert(!std::contiguous_iterator<forward_iterator<int*>>);
static_assert(!std::contiguous_iterator<bidirectional_iterator<int*>>);
static_assert(!std::contiguous_iterator<random_access_iterator<int*>>);
static_assert(std::contiguous_iterator<contiguous_iterator<int*>>);

static_assert(std::contiguous_iterator<int*>);
static_assert(std::contiguous_iterator<int const*>);
static_assert(std::contiguous_iterator<int volatile*>);
static_assert(std::contiguous_iterator<int const volatile*>);

struct simple_contiguous_iterator {
    typedef std::contiguous_iterator_tag    iterator_category;
    typedef int                             value_type;
    typedef int                             element_type;
    typedef std::ptrdiff_t                  difference_type;
    typedef int*                            pointer;
    typedef int&                            reference;
    typedef simple_contiguous_iterator      self;

    simple_contiguous_iterator();

    reference operator*() const;
    pointer operator->() const;
    auto operator<=>(const self&) const = default;

    self& operator++();
    self operator++(int);

    self& operator--();
    self operator--(int);

    self& operator+=(difference_type n);
    self operator+(difference_type n) const;
    friend self operator+(difference_type n, self x);

    self& operator-=(difference_type n);
    self operator-(difference_type n) const;
    difference_type operator-(const self& n) const;

    reference operator[](difference_type n) const;
};

static_assert(std::random_access_iterator<simple_contiguous_iterator>);
static_assert(std::contiguous_iterator<simple_contiguous_iterator>);

struct mismatch_value_iter_ref_t {
    typedef std::contiguous_iterator_tag    iterator_category;
    typedef short                           value_type;
    typedef std::ptrdiff_t                  difference_type;
    typedef int*                            pointer;
    typedef int&                            reference;
    typedef mismatch_value_iter_ref_t       self;

    mismatch_value_iter_ref_t();

    reference operator*() const;
    pointer operator->() const;
    auto operator<=>(const self&) const = default;

    self& operator++();
    self operator++(int);

    self& operator--();
    self operator--(int);

    self& operator+=(difference_type n);
    self operator+(difference_type n) const;
    friend self operator+(difference_type n, self x);

    self& operator-=(difference_type n);
    self operator-(difference_type n) const;
    difference_type operator-(const self& n) const;

    reference operator[](difference_type n) const;
};

static_assert(std::random_access_iterator<mismatch_value_iter_ref_t>);
static_assert(!std::contiguous_iterator<mismatch_value_iter_ref_t>);

struct wrong_iter_reference_t {
    typedef std::contiguous_iterator_tag    iterator_category;
    typedef short                           value_type;
    typedef std::ptrdiff_t                  difference_type;
    typedef int*                            pointer;
    typedef short&                          reference;
    typedef wrong_iter_reference_t          self;

    wrong_iter_reference_t();

    reference operator*() const;
    pointer operator->() const;
    auto operator<=>(const self&) const = default;

    self& operator++();
    self operator++(int);

    self& operator--();
    self operator--(int);

    self& operator+=(difference_type n);
    self operator+(difference_type n) const;
    friend self operator+(difference_type n, self x);

    self& operator-=(difference_type n);
    self operator-(difference_type n) const;
    difference_type operator-(const self& n) const;

    reference operator[](difference_type n) const;
};

static_assert(std::random_access_iterator<wrong_iter_reference_t>);
static_assert(!std::contiguous_iterator<wrong_iter_reference_t>);

struct no_element_type {
    typedef std::contiguous_iterator_tag    iterator_category;
    typedef int                             value_type;
    typedef std::ptrdiff_t                  difference_type;
    typedef int*                            pointer;
    typedef int&                            reference;
    typedef no_element_type                 self;

    no_element_type();

    reference operator*() const;
    pointer operator->() const;
    auto operator<=>(const self&) const = default;

    self& operator++();
    self operator++(int);

    self& operator--();
    self operator--(int);

    self& operator+=(difference_type n);
    self operator+(difference_type n) const;
    friend self operator+(difference_type n, self x);

    self& operator-=(difference_type n);
    self operator-(difference_type n) const;
    difference_type operator-(const self& n) const;

    reference operator[](difference_type n) const;
};

static_assert(std::random_access_iterator<no_element_type>);
static_assert(!std::contiguous_iterator<no_element_type>);

struct to_address_wrong_return_type {
    typedef std::contiguous_iterator_tag    iterator_category;
    typedef int                             value_type;
    typedef int                             element_type;
    typedef std::ptrdiff_t                  difference_type;
    typedef int*                            pointer;
    typedef int&                            reference;
    typedef to_address_wrong_return_type    self;

    to_address_wrong_return_type();

    reference operator*() const;
    pointer operator->() const;
    auto operator<=>(const self&) const = default;

    self& operator++();
    self operator++(int);

    self& operator--();
    self operator--(int);

    self& operator+=(difference_type n);
    self operator+(difference_type n) const;
    friend self operator+(difference_type n, self x);

    self& operator-=(difference_type n);
    self operator-(difference_type n) const;
    difference_type operator-(const self& n) const;

    reference operator[](difference_type n) const;
};

template<>
struct std::pointer_traits<to_address_wrong_return_type> {
  typedef void element_type;
  static void *to_address(to_address_wrong_return_type const&);
};

static_assert(std::random_access_iterator<to_address_wrong_return_type>);
static_assert(!std::contiguous_iterator<to_address_wrong_return_type>);

template<class>
struct template_and_no_element_type {
    typedef std::contiguous_iterator_tag    iterator_category;
    typedef int                             value_type;
    typedef std::ptrdiff_t                  difference_type;
    typedef int*                            pointer;
    typedef int&                            reference;
    typedef template_and_no_element_type    self;

    template_and_no_element_type();

    reference operator*() const;
    pointer operator->() const;
    auto operator<=>(const self&) const = default;

    self& operator++();
    self operator++(int);

    self& operator--();
    self operator--(int);

    self& operator+=(difference_type n);
    self operator+(difference_type n) const;
    friend self operator+(difference_type, self) { return self{}; }

    self& operator-=(difference_type n);
    self operator-(difference_type n) const;
    difference_type operator-(const self& n) const;

    reference operator[](difference_type n) const;
};

// Template param is used instead of element_type.
static_assert(std::random_access_iterator<template_and_no_element_type<int>>);
static_assert(std::contiguous_iterator<template_and_no_element_type<int>>);
