#ifndef FIRSTHEADER
#define FIRSTHEADER

#include "SecondHeader.h" // Just a class which gets in the lazy deserialization chain

#include "stl_map.h"
#include "vector"
typedef std::map<int>::iterator el;

inline void func() {
  std::vector<int>::func();
}

#endif
