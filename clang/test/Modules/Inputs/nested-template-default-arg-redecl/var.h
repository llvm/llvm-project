#ifndef VAR_H
#define VAR_H
struct var_outer {
  template <typename = int>
  static int var;
};
#endif


