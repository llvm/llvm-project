//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <queue>

// template <class InputIterator>
//   priority_queue(InputIterator first, InputIterator last, const Compare& = Compare());
// template <class InputIterator>
//   priority_queue(InputIterator first, InputIterator last, const Compare&, const Container&);
// template <class InputIterator>
//   priority_queue(InputIterator first, InputIterator last, const Compare&, Container&&);
// template <class InputIterator>
//   priority_queue(InputIterator first, InputIterator last, const Alloc&);
// template <class InputIterator>
//   priority_queue(InputIterator first, InputIterator last, const Compare&, const Alloc&);
// template <class InputIterator>
//   priority_queue(InputIterator first, InputIterator last, const Compare&, const Container&, const Alloc&);
// template <class InputIterator>
//   priority_queue(InputIterator first, InputIterator last, const Compare&, Container&&, const Alloc&);

#include <queue>
#include <type_traits>
#include <vector>

// Sanity-check that std::vector is constructible from two ints...
static_assert( std::is_constructible<std::vector<int>,         int*, int*>::value, "");
static_assert( std::is_constructible<std::vector<int>,         int , int >::value, "");

// ...but std::priority_queue is not.
static_assert( std::is_constructible<std::priority_queue<int>, int*, int*>::value, "");
static_assert(!std::is_constructible<std::priority_queue<int>, int , int >::value, "");

static_assert( std::is_constructible<std::priority_queue<int>, int*, int*, std::less<int>>::value, "");
static_assert(!std::is_constructible<std::priority_queue<int>, int , int , std::less<int>>::value, "");

static_assert( std::is_constructible<std::priority_queue<int>, int*, int*, std::less<int>, std::vector<int>>::value, "");
static_assert(!std::is_constructible<std::priority_queue<int>, int , int , std::less<int>, std::vector<int>>::value, "");

static_assert( std::is_constructible<std::priority_queue<int>, int*, int*, std::less<int>, std::vector<int>&>::value, "");
static_assert(!std::is_constructible<std::priority_queue<int>, int , int , std::less<int>, std::vector<int>&>::value, "");

static_assert( std::is_constructible<std::priority_queue<int>, int*, int*, std::allocator<int>>::value, "");
static_assert(!std::is_constructible<std::priority_queue<int>, int , int , std::allocator<int>>::value, "");

static_assert( std::is_constructible<std::priority_queue<int>, int*, int*, std::less<int>, std::allocator<int>>::value, "");
static_assert(!std::is_constructible<std::priority_queue<int>, int , int , std::less<int>, std::allocator<int>>::value, "");

static_assert( std::is_constructible<std::priority_queue<int>, int*, int*, std::less<int>, std::vector<int>, std::allocator<int>>::value, "");
static_assert(!std::is_constructible<std::priority_queue<int>, int , int , std::less<int>, std::vector<int>, std::allocator<int>>::value, "");

static_assert( std::is_constructible<std::priority_queue<int>, int*, int*, std::less<int>, std::vector<int>&, std::allocator<int>>::value, "");
static_assert(!std::is_constructible<std::priority_queue<int>, int , int , std::less<int>, std::vector<int>&, std::allocator<int>>::value, "");
