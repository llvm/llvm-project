#ifndef A_H
#define A_H
template<typename T> struct A { static bool b; };
template<typename T> bool A<T>::b = false;
template<typename T> void *get() { return &(A<T>::b); }
#endif
