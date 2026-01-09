//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <__config>
#include <fstream>
#include <ios>
#include <istream>
#include <ostream>
#include <sstream>
#include <streambuf>

_LIBCPP_BEGIN_NAMESPACE_STD

// Original explicit instantiations provided in the library
template class _LIBCPP_EXPORTED_FROM_ABI basic_ios<char>;
template class _LIBCPP_EXPORTED_FROM_ABI basic_streambuf<char>;
template class _LIBCPP_EXPORTED_FROM_ABI basic_istream<char>;
template class _LIBCPP_EXPORTED_FROM_ABI basic_ostream<char>;
template class _LIBCPP_EXPORTED_FROM_ABI basic_iostream<char>;

#if _LIBCPP_HAS_WIDE_CHARACTERS
template class _LIBCPP_EXPORTED_FROM_ABI basic_ios<wchar_t>;
template class _LIBCPP_EXPORTED_FROM_ABI basic_streambuf<wchar_t>;
template class _LIBCPP_EXPORTED_FROM_ABI basic_istream<wchar_t>;
template class _LIBCPP_EXPORTED_FROM_ABI basic_ostream<wchar_t>;
#endif

// Additional instantiations added later. Whether programs rely on these being
// available is protected by _LIBCPP_AVAILABILITY_HAS_ADDITIONAL_IOSTREAM_EXPLICIT_INSTANTIATIONS_1.
template class _LIBCPP_EXPORTED_FROM_ABI basic_stringbuf<char>;
template class _LIBCPP_EXPORTED_FROM_ABI basic_stringstream<char>;
template class _LIBCPP_EXPORTED_FROM_ABI basic_ostringstream<char>;
template class _LIBCPP_EXPORTED_FROM_ABI basic_istringstream<char>;

#if _LIBCPP_HAS_FILESYSTEM
template class _LIBCPP_EXPORTED_FROM_ABI basic_ifstream<char>;
template class _LIBCPP_EXPORTED_FROM_ABI basic_ofstream<char>;
template class _LIBCPP_EXPORTED_FROM_ABI basic_filebuf<char>;
#endif

// Add more here if needed...

_LIBCPP_END_NAMESPACE_STD
