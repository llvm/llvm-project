//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_SUPPORT_MAKE_TEST_THREAD_H
#define TEST_SUPPORT_MAKE_TEST_THREAD_H

#include <thread>
#include <utility>

namespace support {

template <class F, class ...Args>
std::thread make_test_thread(F&& f, Args&& ...args) {
    return std::thread(std::forward<F>(f), std::forward<Args>(args)...);
}

} // end namespace support

#endif // TEST_SUPPORT_MAKE_TEST_THREAD_H
