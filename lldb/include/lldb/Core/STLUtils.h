//===-- STLUtils.h ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_STLUtils_h_
#define liblldb_STLUtils_h_

#include <string.h>

#include <map>
#include <ostream>
#include <vector>


//----------------------------------------------------------------------
// C string less than compare function object
//----------------------------------------------------------------------
struct CStringCompareFunctionObject {
  bool operator()(const char *s1, const char *s2) const {
    return strcmp(s1, s2) < 0;
  }
};

//----------------------------------------------------------------------
// C string equality function object (binary predicate).
//----------------------------------------------------------------------
struct CStringEqualBinaryPredicate {
  bool operator()(const char *s1, const char *s2) const {
    return strcmp(s1, s2) == 0;
  }
};

//----------------------------------------------------------------------
// Templated type for finding an entry in a std::map<F,S> whose value is equal
// to something
//----------------------------------------------------------------------
template <class F, class S> class ValueEquals {
public:
  ValueEquals(const S &val) : second_value(val) {}

  // Compare the second item
  bool operator()(std::pair<const F, S> elem) {
    return elem.second == second_value;
  }

private:
  S second_value;
};

template <class T>
inline void PrintAllCollectionElements(std::ostream &s, const T &coll,
                                       const char *header_cstr = nullptr,
                                       const char *separator_cstr = " ") {
  typename T::const_iterator pos;

  if (header_cstr)
    s << header_cstr;
  for (pos = coll.begin(); pos != coll.end(); ++pos) {
    s << *pos << separator_cstr;
  }
  s << std::endl;
}

// The function object below can be used to delete a STL container that
// contains C++ object pointers.
//
// Usage: std::for_each(vector.begin(), vector.end(), for_each_delete());

struct for_each_cplusplus_delete {
  template <typename T> void operator()(T *ptr) { delete ptr; }
};

typedef std::vector<std::string> STLStringArray;
typedef std::vector<const char *> CStringArray;

#endif // liblldb_STLUtils_h_
