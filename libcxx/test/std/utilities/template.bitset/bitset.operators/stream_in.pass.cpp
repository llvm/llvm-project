//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-localization

// test:

// template <class charT, class traits, size_t N>
// basic_istream<charT, traits>&
// operator>>(basic_istream<charT, traits>& is, bitset<N>& x);

#include <bitset>
#include <sstream>
#include <cassert>
#include "test_macros.h"

int main(int, char**)
{
    {
        std::istringstream in("01011010");
        std::bitset<8> b;
        in >> b;
        assert(b.to_ulong() == 0x5A);
    }
    {
        // Make sure that input-streaming an empty bitset does not cause the
        // failbit to be set (LWG 3199).
        std::istringstream in("01011010");
        std::bitset<0> b;
        in >> b;
        assert(b.to_string() == "");
        assert(!in.bad());
        assert(!in.fail());
        assert(!in.eof());
        assert(in.good());
    }
#ifndef TEST_HAS_NO_EXCEPTIONS
    {
        std::stringbuf sb;
        std::istream is(&sb);
        is.exceptions(std::ios::failbit);

        bool threw = false;
        try {
            std::bitset<8> b;
            is >> b;
        } catch (std::ios::failure const&) {
            threw = true;
        }

        assert(!is.bad());
        assert(is.fail());
        assert(is.eof());
        assert(threw);
    }
    {
        std::stringbuf sb;
        std::istream is(&sb);
        is.exceptions(std::ios::eofbit);

        bool threw = false;
        try {
            std::bitset<8> b;
            is >> b;
        } catch (std::ios::failure const&) {
            threw = true;
        }

        assert(!is.bad());
        assert(is.fail());
        assert(is.eof());
        assert(threw);
    }
#endif // TEST_HAS_NO_EXCEPTIONS

    return 0;
}
