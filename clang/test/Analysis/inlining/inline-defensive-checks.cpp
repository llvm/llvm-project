// RUN: %clang_analyze_cc1 -analyzer-checker=core -verify %s
// expected-no-diagnostics

extern void __assert_fail (__const char *__assertion, __const char *__file,
                           unsigned int __line, __const char *__function)
__attribute__ ((__noreturn__));
#define assert(expr) \
((expr)  ? (void)(0)  : __assert_fail (#expr, __FILE__, __LINE__, __func__))

class ButterFly {
private:
  ButterFly() { }
public:
	int triggerderef() {
		return 0;
	}
};
ButterFly *getInP();
class X{
	ButterFly *p;
	void setP(ButterFly *inP) {
		if(inP)
      ;
		p = inP;
	};
	void subtest1() {
		ButterFly *inP = getInP();
		setP(inP);
	}
	int subtest2() {
		int c = p->triggerderef(); // no-warning
		return c;
	}
	int test() {
		subtest1();
		return subtest2();
	}
};

typedef const int *Ty;
extern
Ty notNullArg(Ty cf) __attribute__((nonnull));
typedef const void *CFTypeRef;
extern Ty getTyVal();
inline void radar13224271_callee(Ty def, Ty& result ) {
	result = def;
  // Clearly indicates that result cannot be 0 if def is not NULL.
	assert( (result != 0) || (def == 0) );
}
void radar13224271_caller()
{
	Ty value;
	radar13224271_callee(getTyVal(), value );
	notNullArg(value); // no-warning
}

struct Foo {
	int *ptr;
	Foo(int *p)  {
		*p = 1; // no-warning
	}
};
void idc(int *p3) {
  if (p3)
    ;
}
int *retNull() {
  return 0;
}
void test(int *p1, int *p2) {
  idc(p1);
	Foo f(p1);
}

struct Bar {
  int x;
};
void idcBar(Bar *b) {
  if (b)
    ;
}
void testRefToField(Bar *b) {
  idcBar(b);
  int &x = b->x; // no-warning
  x = 5;
}

namespace get_deref_expr_with_cleanups {
struct S {
~S();
};
S *conjure();
// The argument won't be used, but it'll cause cleanups
// to appear around the call site.
S *get_conjured(S _) {
  S *s = conjure();
  if (s) {}
  return s;
}
void test_conjured() {
  S &s = *get_conjured(S()); // no-warning
}
} // namespace get_deref_expr_with_cleanups
