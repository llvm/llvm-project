#ifndef SECONDHEADER
#define SECONDHEADER

#include "vector"

template <class T>
struct Address {};

template <>
struct Address<std::vector<bool>>
    : Address<std::vector<bool>::iterator> {};

#endif
