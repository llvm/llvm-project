//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_STD_INPUTOUTPUT_STRINGSTREAMS_CONCEPTS_H
#define TEST_STD_INPUTOUTPUT_STRINGSTREAMS_CONCEPTS_H

template <typename S, typename T>
concept HasStr = requires(S s, const T& sv) {
  { s.str(sv) };
};

#endif // TEST_STD_INPUTOUTPUT_STRINGSTREAMS_CONCEPTS_H
