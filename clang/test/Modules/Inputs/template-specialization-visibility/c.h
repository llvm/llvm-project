#ifndef C_H
#define C_H
template<typename T> struct S { int n; };
template<typename U> struct T<U>::S { int n; };
template<typename U> enum T<U>::E : int { e };
#endif
