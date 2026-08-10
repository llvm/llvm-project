#ifndef ALIAS_H
#define ALIAS_H
struct alias_outer {
  template <typename = int>
  using alias = int;
};
#endif
