#ifndef FUNC_H
#define FUNC_H
struct func_outer {
  template <typename = int>
  void func();
};
#endif
