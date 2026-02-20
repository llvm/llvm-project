.. title:: clang-tidy - modernize-redundant-void-arg

modernize-redundant-void-arg
============================

Finds and removes redundant ``void`` argument lists.
Works in C++ and in C23 and up.

Examples:
  ===================================  ===========================
  Initial code                         Code with applied fixes
  ===================================  ===========================
  ``int f(void);``                     ``int f();``
  ``int (*f(void))(void);``            ``int (*f())();``
  ``typedef int (*f_t(void))(void);``  ``typedef int (*f_t())();``
  ``void (C::*p)(void);``              ``void (C::*p)();``
  ``C::C(void) {}``                    ``C::C() {}``
  ``C::~C(void) {}``                   ``C::~C() {}``
  ===================================  ===========================
