//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// wbuffer_convert<Codecvt, Elem, Tr>

#include <fstream>
#include <locale>
#include <codecvt>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    {
        std::ofstream bytestream("myfile.txt");
        std::wbuffer_convert<std::codecvt_utf8<wchar_t> > mybuf(bytestream.rdbuf());
        std::wostream mystr(&mybuf);
        mystr << L"Hello" << std::endl;
    }
    {
        std::ifstream bytestream("myfile.txt");
        std::wbuffer_convert<std::codecvt_utf8<wchar_t> > mybuf(bytestream.rdbuf());
        std::wistream mystr(&mybuf);
        std::wstring ws;
        mystr >> ws;
        assert(ws == L"Hello");
    }
    std::remove("myfile.txt");

  return 0;
}
