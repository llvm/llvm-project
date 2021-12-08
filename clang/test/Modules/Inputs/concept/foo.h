#ifndef FOO_H
#define FOO_H

template <class T>
concept Range = requires(T &t) { t.begin(); };

struct A {
public:
  template <Range T>
  using range_type = T;
};

#endif
