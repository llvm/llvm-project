//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/__impl/detail/config.hpp>
#include <sycl/__impl/exception.hpp>

#include <iostream>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

exception_list::size_type exception_list::size() const { return MList.size(); }

exception_list::iterator exception_list::begin() const { return MList.begin(); }

exception_list::iterator exception_list::end() const { return MList.cend(); }

void detail::addAsyncException(exception_list &List,
                               const std::exception_ptr &Exception) {
  List.MList.emplace_back(Exception);
}

void detail::defaultAsyncHandler(exception_list Exceptions) {
  std::cerr << "Default async_handler caught exceptions:";
  for (auto &EIt : Exceptions) {
    try {
      if (EIt) {
        std::rethrow_exception(EIt);
      }
    } catch (const std::exception &E) {
      std::cerr << "\n\t" << E.what();
    }
  }
  std::cerr << std::endl;
  std::terminate();
}

_LIBSYCL_END_NAMESPACE_SYCL
