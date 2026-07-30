// RUN: rm -rf %t
// RUN: %clang_cc1 -std=c++11 -I%S/Inputs/PR27401 -verify %s
// RUN: %clang_cc1 -std=c++11 -fmodules -fmodule-map-file=%S/Inputs/PR27401/module.modulemap -fmodules-cache-path=%t -I%S/Inputs/PR27401 -verify %s
// RUN: %clang_cc1 -std=c++11 -fmodules -fmodule-map-file=%S/Inputs/PR27401/module.modulemap -fmodules-cache-path=%t -I%S/Inputs/PR27401 -verify %s -triple i686-windows

#include "a.h"
#define _LIBCPP_VECTOR
template <class, class _Allocator>
class __vector_base {
protected:
  _Allocator __alloc() const;
  __vector_base(_Allocator);
};

template <class _Tp, class _Allocator = allocator>
class vector : __vector_base<_Tp, _Allocator> {
public:
  vector() noexcept(is_nothrow_default_constructible<_Allocator>::value);
  vector(const vector &);
  vector(vector &&)
      noexcept(is_nothrow_move_constructible<_Allocator>::value);
};

template <class _Tp, class _Allocator>
vector<_Tp, _Allocator>::vector(const vector &__x) : __vector_base<_Tp, _Allocator>(__x.__alloc()) {}

  struct CommentOptions {
    vector<char>  ParseAllComments;
    CommentOptions() {}
  };
  struct PrintingPolicy {
    PrintingPolicy(CommentOptions LO) : LangOpts(LO) {}
    CommentOptions LangOpts;
  };

#include "b.h"
CommentOptions fn1() { return fn1(); }

// expected-no-diagnostics
