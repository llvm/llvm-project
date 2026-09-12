// RUN: %clang_cc1 -verify %s
// expected-no-diagnostics

template<unsigned int SPACE>
char foo_choose() {
  char buffer[SPACE] {__builtin_choose_expr(6, "foo", "boo")};
  return buffer[0];
}

int boro_choose()
{
  int r = foo_choose<10>();
  r += foo_choose<100>();
  return r + foo_choose<4>();
}

template<unsigned int SPACE>
char foo_gen_ext() {
  char buffer[SPACE] {__extension__ (_Generic(0, int: (__extension__ "foo" )))};
  return buffer[0];
}

int boro_gen_ext()
{
  int r = foo_gen_ext<10>();
  r += foo_gen_ext<100>();
  return r + foo_gen_ext<4>();
}

template<unsigned int SPACE>
char foo_paren_predef() {
  char buffer[SPACE] {(((__FILE__)))};
  return buffer[0];
}

int boro_paren_predef()
{
  int r = foo_paren_predef<200000>();
  r += foo_paren_predef<300000>();
  return r + foo_paren_predef<100000>();
}
