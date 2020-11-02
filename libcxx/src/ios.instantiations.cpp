//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "__config"
#include "ios"
#include "istream"
#include "ostream"
#include "streambuf"


_LIBCPP_BEGIN_NAMESPACE_STD

// Original explicit instantiations provided in the library
template class _LIBCPP_CLASS_TEMPLATE_INSTANTIATION_VIS basic_ios<char>;
template class _LIBCPP_CLASS_TEMPLATE_INSTANTIATION_VIS basic_ios<wchar_t>;
template class _LIBCPP_CLASS_TEMPLATE_INSTANTIATION_VIS basic_streambuf<char>;
template class _LIBCPP_CLASS_TEMPLATE_INSTANTIATION_VIS basic_streambuf<wchar_t>;
template class _LIBCPP_CLASS_TEMPLATE_INSTANTIATION_VIS basic_istream<char>;
template class _LIBCPP_CLASS_TEMPLATE_INSTANTIATION_VIS basic_istream<wchar_t>;
template class _LIBCPP_CLASS_TEMPLATE_INSTANTIATION_VIS basic_ostream<char>;
template class _LIBCPP_CLASS_TEMPLATE_INSTANTIATION_VIS basic_ostream<wchar_t>;
template class _LIBCPP_CLASS_TEMPLATE_INSTANTIATION_VIS basic_iostream<char>;

// Add more here if needed...

_LIBCPP_END_NAMESPACE_STD
