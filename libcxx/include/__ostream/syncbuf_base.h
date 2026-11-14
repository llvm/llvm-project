// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___OSTREAM_SYNCBUF_BASE_H
#define _LIBCPP___OSTREAM_SYNCBUF_BASE_H

#include <__config>
#include <__ostream/basic_ostream.h>
#include <streambuf>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 20 && _LIBCPP_HAS_EXPERIMENTAL_SYNCSTREAM

template <class _CharT, class _Traits>
_LIBCPP_HIDE_FROM_ABI basic_ostream<_CharT, _Traits>& emit_on_flush(basic_ostream<_CharT, _Traits>& __os);

template <class _CharT, class _Traits>
_LIBCPP_HIDE_FROM_ABI basic_ostream<_CharT, _Traits>& noemit_on_flush(basic_ostream<_CharT, _Traits>& __os);

template <class _CharT, class _Traits>
_LIBCPP_HIDE_FROM_ABI basic_ostream<_CharT, _Traits>& flush_emit(basic_ostream<_CharT, _Traits>& __os);

template <class _CharT, class _Traits>
class __syncbuf_base : public basic_streambuf<_CharT, _Traits> {
protected:
  _LIBCPP_HIDE_FROM_ABI explicit __syncbuf_base(bool __emit_on_sync) : __emit_on_sync_(__emit_on_sync) {}

private:
  bool __emit_on_sync_;

  _LIBCPP_HIDE_FROM_ABI_VIRTUAL virtual bool __emit(bool __flush) = 0;

  template <class, class, class>
  friend class basic_syncbuf;

  friend basic_ostream<_CharT, _Traits>& emit_on_flush<>(basic_ostream<_CharT, _Traits>&);
  friend basic_ostream<_CharT, _Traits>& noemit_on_flush<>(basic_ostream<_CharT, _Traits>&);
  friend basic_ostream<_CharT, _Traits>& flush_emit<>(basic_ostream<_CharT, _Traits>&);
};

template <class _CharT, class _Traits>
_LIBCPP_HIDE_FROM_ABI basic_ostream<_CharT, _Traits>& emit_on_flush(basic_ostream<_CharT, _Traits>& __os) {
  if (auto* __buf = dynamic_cast<__syncbuf_base<_CharT, _Traits>*>(__os.rdbuf())) {
    __buf->__emit_on_sync_ = true;
  }
  return __os;
}

template <class _CharT, class _Traits>
_LIBCPP_HIDE_FROM_ABI basic_ostream<_CharT, _Traits>& noemit_on_flush(basic_ostream<_CharT, _Traits>& __os) {
  if (auto* __buf = dynamic_cast<__syncbuf_base<_CharT, _Traits>*>(__os.rdbuf())) {
    __buf->__emit_on_sync_ = false;
  }
  return __os;
}

template <class _CharT, class _Traits>
_LIBCPP_HIDE_FROM_ABI basic_ostream<_CharT, _Traits>& flush_emit(basic_ostream<_CharT, _Traits>& __os) {
  __os.flush();
  if (auto* __buf = dynamic_cast<__syncbuf_base<_CharT, _Traits>*>(__os.rdbuf())) {
    // The standard specifies that:
    // After constructing a sentry object, calls buf->emit().
    // If that call returns false, calls os.setstate(ios_base::badbit).
    //
    // syncstream::emit already constructs a sentry
    bool __emit_result = __buf->__emit(false);
    if (!__emit_result) {
      __os.setstate(ios_base::badbit);
    }
  }
  return __os;
}

#endif // _LIBCPP_STD_VER >= 20 && _LIBCPP_HAS_EXPERIMENTAL_SYNCSTREAM

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___OSTREAM_SYNCBUF_BASE_H
