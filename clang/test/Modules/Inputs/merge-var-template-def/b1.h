#ifndef B1_H
#define B1_H
template<typename T> struct A { static bool b; };
template<typename T> bool A<T>::b = false;
template<typename T> void *get() { return &(A<T>::b); }
#include "a.h"
#endif
