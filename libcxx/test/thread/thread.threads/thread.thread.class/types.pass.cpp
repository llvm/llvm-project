//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <thread>

// class thread
// {
// public:
//     typedef pthread_t native_handle_type;
//     ...
// };

#include <thread>
#include <type_traits>

int main()
{
    static_assert((std::is_same<std::thread::native_handle_type, pthread_t>::value), "");
}
