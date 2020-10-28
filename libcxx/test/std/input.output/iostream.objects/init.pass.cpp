//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: libcpp-has-no-stdin, libcpp-has-no-stdout

// Make sure that the iostreams are initialized before everything else.
// This has been an issue when statically linking libc++ in some contexts.
// See https://llvm.org/PR28954 for details.
//
// This test works by checking that std::{cin,cout,cerr} is the same in a
// static object constructor and in the main function. It dumps the memory of
// each stream in the static object constructor and compares it with the memory
// in the main function.
//
// The assumption is that if there are no uses of the stream object (such as
// construction), then its memory must be the same. In the case where the test
// "fails" and we are actually accessing an uninitialized object when we perform
// the memcpy, the behavior is technically undefined (so the test could still
// pass).

#include <cassert>
#include <cstring>
#include <iostream>

struct Checker {
    char *cerr_mem_dump;
    char *cin_mem_dump;
    char *cout_mem_dump;
    char *clog_mem_dump;

    char *wcerr_mem_dump;
    char *wcin_mem_dump;
    char *wcout_mem_dump;
    char *wclog_mem_dump;

    Checker()
        : cerr_mem_dump(new char[sizeof(std::cerr)])
        , cin_mem_dump(new char[sizeof(std::cin)])
        , cout_mem_dump(new char[sizeof(std::cout)])
        , clog_mem_dump(new char[sizeof(std::clog)])

        , wcerr_mem_dump(new char[sizeof(std::wcerr)])
        , wcin_mem_dump(new char[sizeof(std::wcin)])
        , wcout_mem_dump(new char[sizeof(std::wcout)])
        , wclog_mem_dump(new char[sizeof(std::wclog)])
     {
        std::memcpy(cerr_mem_dump, (char*)&std::cerr, sizeof(std::cerr));
        std::memcpy(cin_mem_dump, (char*)&std::cin, sizeof(std::cin));
        std::memcpy(cout_mem_dump, (char*)&std::cout, sizeof(std::cout));
        std::memcpy(clog_mem_dump, (char*)&std::clog, sizeof(std::clog));

        std::memcpy(wcerr_mem_dump, (char*)&std::wcerr, sizeof(std::wcerr));
        std::memcpy(wcin_mem_dump, (char*)&std::wcin, sizeof(std::wcin));
        std::memcpy(wcout_mem_dump, (char*)&std::wcout, sizeof(std::wcout));
        std::memcpy(wclog_mem_dump, (char*)&std::wclog, sizeof(std::wclog));
    }

    ~Checker() {
        delete[] cerr_mem_dump;
        delete[] cin_mem_dump;
        delete[] cout_mem_dump;
        delete[] clog_mem_dump;

        delete[] wcerr_mem_dump;
        delete[] wcin_mem_dump;
        delete[] wcout_mem_dump;
        delete[] wclog_mem_dump;
    }
};

static Checker check;

int main(int, char**) {
    assert(std::memcmp(check.cerr_mem_dump, (char const*)&std::cerr, sizeof(std::cerr)) == 0);
    assert(std::memcmp(check.cin_mem_dump, (char const*)&std::cin, sizeof(std::cin)) == 0);
    assert(std::memcmp(check.cout_mem_dump, (char const*)&std::cout, sizeof(std::cout)) == 0);
    assert(std::memcmp(check.clog_mem_dump, (char const*)&std::clog, sizeof(std::clog)) == 0);

    assert(std::memcmp(check.wcerr_mem_dump, (char const*)&std::wcerr, sizeof(std::wcerr)) == 0);
    assert(std::memcmp(check.wcin_mem_dump, (char const*)&std::wcin, sizeof(std::wcin)) == 0);
    assert(std::memcmp(check.wcout_mem_dump, (char const*)&std::wcout, sizeof(std::wcout)) == 0);
    assert(std::memcmp(check.wclog_mem_dump, (char const*)&std::wclog, sizeof(std::wclog)) == 0);
    return 0;
}
