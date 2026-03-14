// Header for PCH test cxx-required-decls.cpp

struct S {
  S();
};

static S globS;

extern int ext_foo;
static int bar = ++ext_foo;

int glob_var;
