//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11
// UNSUPPORTED: no-exceptions
// UNSUPPORTED: libcpp-has-no-localization

#include <cstddef>
#include <cstdint>
#include <regex>
#include <string>

#include "fuzz.h"

template <std::regex_constants::syntax_option_type Syntax>
static int regex_test(const std::uint8_t *data, std::size_t size) {
    if (size == 0)
        return 0;

    std::string s((const char *)data, size);
    std::regex re;
    try {
        re.assign(s, Syntax);
    } catch (std::regex_error &) {
        // the data represents an invalid regex, ignore this test case
        return 0;
    }

    auto match = std::regex_match(s, re);
    (void)match;
    return 0; // always pretend we succeeded -- we're only looking for crashes
}

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t *data, std::size_t size) {
    return regex_test<std::regex_constants::awk>(data, size)        ||
           regex_test<std::regex_constants::basic>(data, size)      ||
           regex_test<std::regex_constants::ECMAScript>(data, size) ||
           regex_test<std::regex_constants::egrep>(data, size)      ||
           regex_test<std::regex_constants::extended>(data, size)   ||
           regex_test<std::regex_constants::grep>(data, size);
}
