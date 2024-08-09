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

// If we're building for a lower __ANDROID_API__, the Bionic versioner will
// omit the function declarations from fdsan.h. We might be running on a newer
// API level, though, so declare the API function here using weak.
#if defined(__BIONIC__)
#include <android/fdsan.h>
enum android_fdsan_error_level android_fdsan_set_error_level(enum android_fdsan_error_level new_level)
    __attribute__((weak));
#endif

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
#if defined(__BIONIC__)
    // Starting with Android API 30+, Bionic's fdsan aborts a process that
    // attempts to close a file descriptor belonging to something else. Disable
    // fdsan to allow closing the FD belonging to std::filebuf's FILE*.
    if (android_fdsan_set_error_level != nullptr)
        android_fdsan_set_error_level(ANDROID_FDSAN_ERROR_LEVEL_DISABLED);
#endif
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
