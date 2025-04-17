// RUN: mkdir -p %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -std=c++20 -x c++-header %t/bad-header-unit.h \
// RUN:  -emit-header-unit -o %t/bad-header-unit.pcm -verify

//--- bad-header-unit.h

inline int ok_foo () { return 0;}

static int ok_bar ();

int ok_decl ();

int bad_def () { return 2;}  // expected-error {{non-inline external definitions are not permitted in C++ header units}}

inline int ok_inline_var = 1;

static int ok_static_var;

int ok_var_decl;

int bad_var_definition = 3;  // expected-error {{non-inline external definitions are not permitted in C++ header units}}

/* The cases below should compile without diagnostics.  */

class A {
public:
    // This is a declaration instead of definition.
    static const int value = 43; 
};

void deleted_fn_ok (void) = delete;

struct S {
   ~S() noexcept(false) = default;
private:
  S(S&);
};
S::S(S&) = default;

template <class _X>
_X tmpl_var_ok_0 = static_cast<_X>(-1);

template <typename _T>
constexpr _T tmpl_var_ok_1 = static_cast<_T>(42);

inline int a = tmpl_var_ok_1<int>;

template <typename _Tp,
          template <typename> class _T>
constexpr int tmpl_var_ok_2 = _T<_Tp>::value ? 42 : 6174 ;

template<class _Ep>
int tmpl_OK (_Ep) { return 0; }

template <class _T1>
bool
operator==(_T1& , _T1& ) { return false; }

constexpr long one_k = 1000L;

template <class ..._Args>
void* tmpl_fn_ok
(_Args ...__args) { return nullptr; }

inline int foo (int a) {
  return tmpl_OK (a);
}

template <typename T> struct S2 { static int v; };
template <typename T> int S2<T>::v = 10;

template <typename T> bool b() {
    bool b1 = S2<T>::v == 10;
    return b1 && true;
}

inline bool B = b<int>();
