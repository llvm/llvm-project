//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: gcc-10

// REQUIRES: locale.en_US.UTF-8

#include <iterator>

#include <istream>
#include <ostream>

static_assert(std::__iterator_traits_detail::__cpp17_iterator<std::istream_iterator<int, std::istream> >);
static_assert(std::__iterator_traits_detail::__cpp17_iterator<std::istreambuf_iterator<int, std::istream> >);
static_assert(std::__iterator_traits_detail::__cpp17_iterator<std::ostream_iterator<int, std::ostream> >);
static_assert(std::__iterator_traits_detail::__cpp17_iterator<std::ostreambuf_iterator<int, std::ostream> >);

static_assert(std::__iterator_traits_detail::__cpp17_input_iterator<std::istream_iterator<int, std::istream> >);
static_assert(std::__iterator_traits_detail::__cpp17_input_iterator<std::istreambuf_iterator<int, std::istream> >);
static_assert(!std::__iterator_traits_detail::__cpp17_input_iterator<std::ostream_iterator<int, std::ostream> >);
static_assert(!std::__iterator_traits_detail::__cpp17_input_iterator<std::ostreambuf_iterator<int, std::ostream> >);

// This is because the legacy iterator concepts don't care about iterator_category
static_assert(std::__iterator_traits_detail::__cpp17_forward_iterator<std::istream_iterator<int, std::istream> >);

static_assert(!std::__iterator_traits_detail::__cpp17_forward_iterator<std::istreambuf_iterator<int, std::istream> >);
static_assert(!std::__iterator_traits_detail::__cpp17_forward_iterator<std::ostream_iterator<int, std::ostream> >);
static_assert(!std::__iterator_traits_detail::__cpp17_forward_iterator<std::ostreambuf_iterator<int, std::ostream> >);

static_assert(
    !std::__iterator_traits_detail::__cpp17_bidirectional_iterator<std::istream_iterator<int, std::istream> >);
static_assert(
    !std::__iterator_traits_detail::__cpp17_bidirectional_iterator<std::istreambuf_iterator<int, std::istream> >);
static_assert(
    !std::__iterator_traits_detail::__cpp17_bidirectional_iterator<std::ostream_iterator<int, std::ostream> >);
static_assert(
    !std::__iterator_traits_detail::__cpp17_bidirectional_iterator<std::ostreambuf_iterator<int, std::ostream> >);

static_assert(
    !std::__iterator_traits_detail::__cpp17_random_access_iterator<std::istream_iterator<int, std::istream> >);
static_assert(
    !std::__iterator_traits_detail::__cpp17_random_access_iterator<std::istreambuf_iterator<int, std::istream> >);
static_assert(
    !std::__iterator_traits_detail::__cpp17_random_access_iterator<std::ostream_iterator<int, std::ostream> >);
static_assert(
    !std::__iterator_traits_detail::__cpp17_random_access_iterator<std::ostreambuf_iterator<int, std::ostream> >);
