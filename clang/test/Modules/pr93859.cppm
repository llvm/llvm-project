// Reduced from https://github.com/llvm/llvm-project/issues/93859
//
// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/reduced_std.cppm -emit-reduced-module-interface -o %t/reduced_std.pcm
// RUN: %clang_cc1 -std=c++20 %t/Misc.cppm -emit-reduced-module-interface -o %t/Misc.pcm \
// RUN:     -fprebuilt-module-path=%t
// RUN: %clang_cc1 -std=c++20 %t/Instance.cppm -emit-reduced-module-interface -o %t/Instance.pcm \
// RUN:     -fprebuilt-module-path=%t
// RUN: %clang_cc1 -std=c++20 %t/Device.cppm -emit-reduced-module-interface -o %t/Device.pcm \
// RUN:     -fprebuilt-module-path=%t
// RUN: %clang_cc1 -std=c++20 %t/Overlay.cppm -emit-reduced-module-interface -o %t/Overlay.pcm \
// RUN:     -fprebuilt-module-path=%t
// RUN: %clang_cc1 -std=c++20 %t/App.cppm -emit-module-interface -o /dev/null \
// RUN:     -fexperimental-modules-reduced-bmi -fmodule-output=%t/App.pcm \
// RUN:     -fprebuilt-module-path=%t
// RUN: %clang_cc1 -std=c++20 %t/test.cc -fsyntax-only -verify \
// RUN:     -fprebuilt-module-path=%t

//--- header.h
namespace std {

template <class _T1, class _T2>
struct pair
{
  _T1 first;
  _T2 second;

  constexpr pair()
      : first(), second() {}

  constexpr pair(_T1 const& __t1, _T2 const& __t2)
      : first(__t1), second(__t2) {}
};

template <class _T1, class _T2>
pair(_T1, _T2) -> pair<_T1, _T2>;

template <class _Tp>
class __tree_const_iterator {
public:
  template <class>
  friend class __tree;
};

template <class _Tp>
class __tree {
public:
  typedef _Tp value_type;
  typedef __tree_const_iterator<value_type> const_iterator;

  template <class, class, class, class>
  friend class map;
};

template <class _Key>
class set {
public:
  typedef __tree<_Key> __base;

  typedef typename __base::const_iterator iterator;

  set() {}

  pair<iterator, bool>
  insert(const _Key& __v);
};

template <class _InputIterator, class _OutputIterator>
inline constexpr _OutputIterator
copy(_InputIterator __first, _InputIterator __last, _OutputIterator __result) {
  return pair{__first, __last}.second;
}

}

//--- reduced_std.cppm
module;
#include "header.h"
export module reduced_std;

export namespace std {
    using std::set;
    using std::copy;
}

//--- Misc.cppm
export module Misc;
import reduced_std;

export void check_result(int res) {
    std::set<char> extensions;
    extensions.insert('f');
}

//--- Instance.cppm
export module Instance;
import reduced_std;

export class Instance {
public:
    Instance() {
        std::set<const char*> extensions;
        extensions.insert("foo");
    }
};

//--- Device.cppm
export module Device;
import reduced_std;
import Instance;
import Misc;

std::set<int> wtf_set;

//--- Overlay.cppm
export module Overlay;

import reduced_std;
import Device;

void overlay_vector_use() {
    std::set<int> nums;
    nums.insert(1);
}

//--- App.cppm
module;
#include "header.h"
export module App;
import Overlay;

std::set<float> fs;

//--- test.cc
// expected-no-diagnostics
import reduced_std;
import App;

void render() {
    unsigned *oidxs = nullptr;
    unsigned idxs[] = {0, 1, 2, 0, 2, 3};
    std::copy(idxs, idxs + 6, oidxs);
}
