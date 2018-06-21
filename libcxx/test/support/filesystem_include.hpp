#ifndef TEST_SUPPORT_FILESYSTEM_INCLUDE_HPP
#define TEST_SUPPORT_FILESYSTEM_INCLUDE_HPP

#include <ciso646>
// Test against std::filesystem for STL's other than libc++
#ifndef _LIBCPP_VERSION
#define TEST_INCLUDE_STD_FILESYSTEM
#endif

#ifdef TEST_INCLUDE_STD_FILESYSTEM
#include <filesystem>
namespace fs = std::filesystem;
#else
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

#endif
