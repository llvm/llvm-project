#include <vector>

template<typename T> struct A { static bool b; };
template<typename T> bool A<T>::b;
