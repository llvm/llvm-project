#ifndef _T_H_
#define _T_H_

extern int some_val;

namespace NS {
  class C {
    void method_decl();
    int method_def1() { ++some_val; return undef_val1; }
    inline int method_def2();
  };
}

inline int NS::C::method_def2() {
  ++some_val; return undef_val2;
}

static inline int foo1() {
  ++some_val; return undef_val3;
}

#ifdef BLAH

static inline int foo2() {
  ++some_val; return undef_val4;
}

#endif

#endif
