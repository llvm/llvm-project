// RUN: %clang_cc1 -verify -Wfenv-access %s

typedef struct {} fenv_t;
typedef unsigned short int fexcept_t;

int feclearexcept(int excepts);
int fegetexceptflag(fexcept_t *flagp, int excepts);
int feraiseexcept(int excepts);
int fesetexceptflag(const fexcept_t *flagp, int excepts);
int fetestexcept(int excepts);
int fegetround(void);
int fesetround(int rounding_mode);
int fegetenv(fenv_t *envp);
int feholdexcept(fenv_t *envp);
int fesetenv(const fenv_t *envp);
int feupdateenv(const fenv_t *envp);

// expected-no-diagnostics
void test_fenv_access_unevaluated() {
  decltype(::feclearexcept) a;
  decltype(::fegetexceptflag) b;
  decltype(::feraiseexcept) c;
  decltype(::fesetexceptflag) d;
  decltype(::fetestexcept) e;
  decltype(::fegetround) f;
  decltype(::fesetround) g;
  decltype(::fegetenv) h;
  decltype(::feholdexcept) i;
  decltype(::fesetenv) j;
  decltype(::feupdateenv) k;
}
