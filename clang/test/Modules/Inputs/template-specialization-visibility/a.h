#ifndef A_H
#define A_H
template<typename T> struct S;
template<typename U> struct T {
  struct S;
  enum E : int;
};
#endif
