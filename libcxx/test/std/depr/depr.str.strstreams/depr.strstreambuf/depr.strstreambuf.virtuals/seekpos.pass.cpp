//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <strstream>

// class strstreambuf

// pos_type seekpos(pos_type sp,
//                  ios_base::openmode which = ios_base::in | ios_base::out);

#include <strstream>
#include <cassert>

int main()
{
    {
        char buf[] = "0123456789";
        std::strstreambuf sb(buf, 0);
        assert(sb.pubseekpos(3, std::ios_base::out) == -1);
        assert(sb.pubseekpos(3, std::ios_base::in | std::ios_base::out) == -1);
        assert(sb.pubseekpos(3, std::ios_base::in) == 3);
        assert(sb.sgetc() == '3');
    }
    {
        char buf[] = "0123456789";
        std::strstreambuf sb(buf, 0, buf);
        assert(sb.pubseekpos(3, std::ios_base::in) == 3);
        assert(sb.pubseekpos(3, std::ios_base::out | std::ios_base::in) == 3);
        assert(sb.pubseekpos(3, std::ios_base::out) == 3);
        assert(sb.sputc('a') == 'a');
        assert(sb.str() == std::string("012a456789"));
    }
}
