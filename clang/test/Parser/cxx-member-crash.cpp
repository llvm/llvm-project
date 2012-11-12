// RUN: %clang_cc1 -fsyntax-only %s 2>&1|FileCheck %s

// <rdar://problem/9221993>

// We only care to chek whether the compiler crashes; the actual
// diagnostics are uninteresting.
// CHECK: 8 errors generated.
template<class _CharT>     struct char_traits;
template<typename _CharT, typename _Traits = char_traits<_CharT> >     class basic_ios;
template<typename _CharT, typename _Traits = char_traits<_CharT> >     class ostreambuf_iterator;
template<typename _CharT, typename _InIter = istreambuf_iterator<_CharT> >     class num_get;
template<typename _CharT, typename _Traits>     class basic_ostream : virtual public basic_ios<_CharT, _Traits>     {
  template<typename _CharT, typename _InIter>     _InIter     num_get<_CharT, _InIter>::     _M_extract_float(_InIter __beg, _InIter __end, ios_base& __io,        ios_base::iostate& __err, string& __xtrc) const     {
    const bool __plus = __c == __lit[__num_base::_S_iplus];
    if ((__plus || __c == __lit[__num_base::_S_iminus])        && !(__lc->_M_use_grouping && __c == __lc->_M_thousands_sep)        && !(__c == __lc->_M_decimal_point))      {
