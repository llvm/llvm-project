//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS -D_LIBCPP_ENABLE_CXX26_REMOVED_CODECVT -D_LIBCPP_ENABLE_CXX26_REMOVED_WSTRING_CONVERT

// wbuffer_convert<Codecvt, Elem, Tr>

// XFAIL: no-wide-characters

#include <cassert>
#include <codecvt>
#include <locale>
#include <sstream>

int main(int, char**) {
    std::string storage;
    {
        std::ostringstream bytestream;
        std::wbuffer_convert<std::codecvt_utf8<wchar_t> > mybuf(bytestream.rdbuf());
        std::wostream mystr(&mybuf);
        mystr << L"Hello" << std::endl;
        storage = bytestream.str();
    }
    {
        std::istringstream bytestream(storage);
        std::wbuffer_convert<std::codecvt_utf8<wchar_t> > mybuf(bytestream.rdbuf());
        std::wistream mystr(&mybuf);
        std::wstring ws;
        mystr >> ws;
        assert(ws == L"Hello");
    }

    return 0;
}
