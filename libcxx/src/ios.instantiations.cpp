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
_LIBCPP_BEGIN_EXPLICIT_ABI_ANNOTATIONS

#if _LIBCPP_HAS_FILESYSTEM
// This is provided for ABI compatiblity with programs compiled against old headers.
// TODO: Is this actually required? If so, this should be guarded with
// `_LIBCPP_AVAILABILITY_MINIMUM_HEADER_VERSION < 24`. Otherwise this should be removed.
template <class _CharT, class _Traits>
bool basic_filebuf<_CharT, _Traits>::__read_mode() {
  if (!(__cm_ & ios_base::in)) {
    this->setp(nullptr, nullptr);
    if (__always_noconv_)
      this->setg((char_type*)__extbuf_, (char_type*)__extbuf_ + __ebs_, (char_type*)__extbuf_ + __ebs_);
    else
      this->setg(__intbuf_, __intbuf_ + __ibs_, __intbuf_ + __ibs_);
    __cm_ = ios_base::in;
    return true;
  }
  return false;
}
#endif

// Original explicit instantiations provided in the library
template class _LIBCPP_CLASS_TEMPLATE_INSTANTIATION_VIS basic_ios<char>;
template class _LIBCPP_CLASS_TEMPLATE_INSTANTIATION_VIS basic_streambuf<char>;
template class _LIBCPP_CLASS_TEMPLATE_INSTANTIATION_VIS basic_istream<char>;
template class _LIBCPP_CLASS_TEMPLATE_INSTANTIATION_VIS basic_ostream<char>;
template class _LIBCPP_CLASS_TEMPLATE_INSTANTIATION_VIS basic_iostream<char>;

#if _LIBCPP_HAS_WIDE_CHARACTERS
template class _LIBCPP_CLASS_TEMPLATE_INSTANTIATION_VIS basic_ios<wchar_t>;
template class _LIBCPP_CLASS_TEMPLATE_INSTANTIATION_VIS basic_streambuf<wchar_t>;
template class _LIBCPP_CLASS_TEMPLATE_INSTANTIATION_VIS basic_istream<wchar_t>;
template class _LIBCPP_CLASS_TEMPLATE_INSTANTIATION_VIS basic_ostream<wchar_t>;
#endif

// Additional instantiations added later. Whether programs rely on these being
// available is protected by _LIBCPP_AVAILABILITY_HAS_ADDITIONAL_IOSTREAM_EXPLICIT_INSTANTIATIONS_1.
template class _LIBCPP_CLASS_TEMPLATE_INSTANTIATION_VIS basic_stringbuf<char>;
template class _LIBCPP_CLASS_TEMPLATE_INSTANTIATION_VIS basic_stringstream<char>;
template class _LIBCPP_CLASS_TEMPLATE_INSTANTIATION_VIS basic_ostringstream<char>;
template class _LIBCPP_CLASS_TEMPLATE_INSTANTIATION_VIS basic_istringstream<char>;

#if _LIBCPP_HAS_FILESYSTEM || _LIBCPP_HAS_FSTREAM
template class _LIBCPP_CLASS_TEMPLATE_INSTANTIATION_VIS basic_ifstream<char>;
template class _LIBCPP_CLASS_TEMPLATE_INSTANTIATION_VIS basic_ofstream<char>;
template class _LIBCPP_CLASS_TEMPLATE_INSTANTIATION_VIS basic_filebuf<char>;
#endif

// Add more here if needed...

_LIBCPP_END_EXPLICIT_ABI_ANNOTATIONS
_LIBCPP_END_NAMESPACE_STD
