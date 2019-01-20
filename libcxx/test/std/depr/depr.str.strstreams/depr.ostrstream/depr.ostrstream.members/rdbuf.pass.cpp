//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <strstream>

// class ostrstream

// strstreambuf* rdbuf() const;

#include <strstream>
#include <cassert>

int main()
{
    {
        char buf[] = "123 4.5 dog";
        const std::ostrstream out(buf, 0);
        std::strstreambuf* sb = out.rdbuf();
        assert(sb->sputc('a') == 'a');
        assert(buf == std::string("a23 4.5 dog"));
    }
}
