//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <fstream>

// basic_filebuf<charT,traits>* close();

#include <fstream>
#include <cassert>
#if defined(__unix__)
#include <fcntl.h>
#include <unistd.h>
#endif
#include "test_macros.h"
#include "platform_support.h"

int main(int, char**)
{
    std::string temp = get_temp_file_name();
    {
        std::filebuf f;
        assert(!f.is_open());
        assert(f.open(temp.c_str(), std::ios_base::out) != 0);
        assert(f.is_open());
        assert(f.close() != nullptr);
        assert(!f.is_open());
        assert(f.close() == nullptr);
        assert(!f.is_open());
    }
#if defined(__unix__)
    {
        std::filebuf f;
        assert(!f.is_open());
        // Use open directly to get the file descriptor.
        int fd = open(temp.c_str(), O_RDWR);
        assert(fd >= 0);
        // Use the internal method to create filebuf from the file descriptor.
        assert(f.__open(fd, std::ios_base::out) != 0);
        assert(f.is_open());
        // Close the file descriptor directly to force filebuf::close to fail.
        assert(close(fd) == 0);
        // Ensure that filebuf::close handles the failure.
        assert(f.close() == nullptr);
        assert(!f.is_open());
        assert(f.close() == nullptr);
    }
#endif
    std::remove(temp.c_str());

    return 0;
}
