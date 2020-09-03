// RUN: %check_clang_tidy %s modernize-redundant-void-arg %t

#define NULL 0

int foo();

void bar();

void bar2();

extern "C" void ecfoo(void);

extern "C" void ecfoo(void) {
}

extern int i;

int j = 1;

int foo(void) {
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: redundant void argument list in function definition [modernize-redundant-void-arg]
// CHECK-FIXES: {{^}}int foo() {{{$}}
    return 0;
}

typedef unsigned int my_uint;

typedef void my_void;

// A function taking void and returning a pointer to function taking void
// and returning int.
int (*returns_fn_void_int(void))(void);
// CHECK-MESSAGES: :[[@LINE-1]]:27: warning: {{.*}} in function declaration
// CHECK-MESSAGES: :[[@LINE-2]]:34: warning: {{.*}} in function declaration
// CHECK-FIXES: {{^}}int (*returns_fn_void_int())();{{$}}

typedef int (*returns_fn_void_int_t(void))(void);
// CHECK-MESSAGES: :[[@LINE-1]]:37: warning: {{.*}} in typedef
// CHECK-MESSAGES: :[[@LINE-2]]:44: warning: {{.*}} in typedef
// CHECK-FIXES: {{^}}typedef int (*returns_fn_void_int_t())();{{$}}

// Should work for type aliases as well as typedef.
using returns_fn_void_int_t2 = int (*(void))(void);
// CHECK-MESSAGES: :[[@LINE-1]]:39: warning: {{.*}} in type alias
// CHECK-MESSAGES: :[[@LINE-2]]:46: warning: {{.*}} in type alias
// CHECK-FIXES: {{^}}using returns_fn_void_int_t2 = int (*())();{{$}}

int (*returns_fn_void_int(void))(void) {
// CHECK-MESSAGES: :[[@LINE-1]]:27: warning: {{.*}} in function definition
// CHECK-MESSAGES: :[[@LINE-2]]:34: warning: {{.*}} in function definition
// CHECK-FIXES: {{^}}int (*returns_fn_void_int())() {{{$}}
  return nullptr;
}

// A function taking void and returning a pointer to a function taking void
// and returning a pointer to a function taking void and returning void.
void (*(*returns_fn_returns_fn_void_void(void))(void))(void);
// CHECK-MESSAGES: :[[@LINE-1]]:42: warning: {{.*}} in function declaration
// CHECK-MESSAGES: :[[@LINE-2]]:49: warning: {{.*}} in function declaration
// CHECK-MESSAGES: :[[@LINE-3]]:56: warning: {{.*}} in function declaration
// CHECK-FIXES: {{^}}void (*(*returns_fn_returns_fn_void_void())())();{{$}}

typedef void (*(*returns_fn_returns_fn_void_void_t(void))(void))(void);
// CHECK-MESSAGES: :[[@LINE-1]]:52: warning: {{.*}} in typedef
// CHECK-MESSAGES: :[[@LINE-2]]:59: warning: {{.*}} in typedef
// CHECK-MESSAGES: :[[@LINE-3]]:66: warning: {{.*}} in typedef
// CHECK-FIXES: {{^}}typedef void (*(*returns_fn_returns_fn_void_void_t())())();{{$}}

void (*(*returns_fn_returns_fn_void_void(void))(void))(void) {
// CHECK-MESSAGES: :[[@LINE-1]]:42: warning: {{.*}} in function definition
// CHECK-MESSAGES: :[[@LINE-2]]:49: warning: {{.*}} in function definition
// CHECK-MESSAGES: :[[@LINE-3]]:56: warning: {{.*}} in function definition
// CHECK-FIXES: {{^}}void (*(*returns_fn_returns_fn_void_void())())() {{{$}}
    return nullptr;
}

void bar(void) {
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: {{.*}} in function definition
// CHECK-FIXES: {{^}}void bar() {{{$}}
}

void op_fn(int i) {
}

class gronk {
public:
  gronk();
  ~gronk();

    void foo();
    void bar();
    void bar2
        ();
    void operation(int i) { }

private:
    int m_i;
    int *m_pi;
    float m_f;
    float *m_pf;
    double m_d;
    double *m_pd;

    void (*f1)(void);
    // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: {{.*}} in field declaration
    // CHECK-FIXES: {{^    }}void (*f1)();{{$}}

  void (*op)(int i);

  void (gronk::*p1)(void);
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: {{.*}} in field declaration
  // CHECK-FIXES: {{^  }}void (gronk::*p1)();{{$}}

  int (gronk::*p_mi);

  void (gronk::*p2)(int);

  void (*(*returns_fn_returns_fn_void_void(void))(void))(void);
  // CHECK-MESSAGES: :[[@LINE-1]]:44: warning: {{.*}} in function declaration
  // CHECK-MESSAGES: :[[@LINE-2]]:51: warning: {{.*}} in function declaration
  // CHECK-MESSAGES: :[[@LINE-3]]:58: warning: {{.*}} in function declaration
  // CHECK-FIXES: {{^}}  void (*(*returns_fn_returns_fn_void_void())())();{{$}}

  void (*(*(gronk::*returns_fn_returns_fn_void_void_mem)(void))(void))(void);
  // CHECK-MESSAGES: :[[@LINE-1]]:58: warning: {{.*}} in field declaration
  // CHECK-MESSAGES: :[[@LINE-2]]:65: warning: {{.*}} in field declaration
  // CHECK-MESSAGES: :[[@LINE-3]]:72: warning: {{.*}} in field declaration
  // CHECK-FIXES: {{^}}  void (*(*(gronk::*returns_fn_returns_fn_void_void_mem)())())();{{$}}
};

int i;
int *pi;
void *pv = (void *) pi;
float f;
float *fi;
double d;
double *pd;

void (*f1)(void);
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: {{.*}} in variable declaration
// CHECK-FIXES: {{^}}void (*f1)();{{$}}

void (*f2)(void) = nullptr;
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: {{.*}} in variable declaration with initializer
// CHECK-FIXES: {{^}}void (*f2)() = nullptr;{{$}}

void (*f2b)(void)(nullptr);
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: {{.*}} in variable declaration with initializer
// CHECK-FIXES: {{^}}void (*f2b)()(nullptr);{{$}}

void (*f2c)(void){nullptr};
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: {{.*}} in variable declaration with initializer
// CHECK-FIXES: {{^}}void (*f2c)(){nullptr};{{$}}

void (*f2d)(void) = NULL;
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: {{.*}} in variable declaration with initializer
// CHECK-FIXES: {{^}}void (*f2d)() = NULL;{{$}}

void (*f2e)(void)(NULL);
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: {{.*}} in variable declaration with initializer
// CHECK-FIXES: {{^}}void (*f2e)()(NULL);{{$}}

void (*f2f)(void){NULL};
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: {{.*}} in variable declaration with initializer
// CHECK-FIXES: {{^}}void (*f2f)(){NULL};{{$}}

void (*f3)(void) = bar;
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: {{.*}} in variable declaration with initializer
// CHECK-FIXES: {{^}}void (*f3)() = bar;{{$}}

void (*o1)(int i);
void (*o2)(int i) = nullptr;
void (*o3)(int i)(nullptr);
void (*o4)(int i){nullptr};
void (*o5)(int i) = NULL;
void (*o6)(int i)(NULL);
void (*o7)(int i){NULL};
void (*o8)(int i) = op_fn;

void (*fa)();

void (*fb)() = nullptr;

void (*fc)() = bar;

typedef void (function_ptr)(void);
// CHECK-MESSAGES: :[[@LINE-1]]:29: warning: {{.*}} in typedef
// CHECK-FIXES: {{^}}typedef void (function_ptr)();{{$}}

// intentionally not LLVM style to check preservation of whitespace
typedef void (function_ptr2)
    (
        void
    );
// CHECK-MESSAGES: :[[@LINE-2]]:9: warning: {{.*}} in typedef
// CHECK-FIXES:      {{^typedef void \(function_ptr2\)$}}
// CHECK-FIXES-NEXT: {{^    \($}}
// CHECK-FIXES-NEXT: {{^        $}}
// CHECK-FIXES-NEXT: {{^    \);$}}

// intentionally not LLVM style to check preservation of whitespace
typedef
void
(
*
(
*
returns_fn_returns_fn_void_void_t2
(
void
)
)
(
void
)
)
(
void
)
;
// CHECK-MESSAGES: :[[@LINE-11]]:1: warning: {{.*}} in typedef
// CHECK-MESSAGES: :[[@LINE-8]]:1: warning: {{.*}} in typedef
// CHECK-MESSAGES: :[[@LINE-5]]:1: warning: {{.*}} in typedef
// CHECK-FIXES:      {{^typedef$}}
// CHECK-FIXES-NEXT: {{^void$}}
// CHECK-FIXES-NEXT: {{^\($}}
// CHECK-FIXES-NEXT: {{^\*$}}
// CHECK-FIXES-NEXT: {{^\($}}
// CHECK-FIXES-NEXT: {{^\*$}}
// CHECK-FIXES-NEXT: {{^returns_fn_returns_fn_void_void_t2$}}
// CHECK-FIXES-NEXT: {{^\($}}
// CHECK-FIXES-NOT:  {{[^ ]}}
// CHECK-FIXES:      {{^\)$}}
// CHECK-FIXES-NEXT: {{^\)$}}
// CHECK-FIXES-NEXT: {{^\($}}
// CHECK-FIXES-NOT:  {{[^ ]}}
// CHECK-FIXES:      {{^\)$}}
// CHECK-FIXES-NEXT: {{^\)$}}
// CHECK-FIXES-NEXT: {{^\($}}
// CHECK-FIXES-NOT:  {{[^ ]}}
// CHECK-FIXES:      {{^\)$}}
// CHECK-FIXES-NEXT: {{^;$}}


void (gronk::*p1)(void);
// CHECK-MESSAGES: :[[@LINE-1]]:19: warning: {{.*}} in variable declaration
// CHECK-FIXES: {{^}}void (gronk::*p1)();{{$}}

void (gronk::*p2)(void) = &gronk::foo;
// CHECK-MESSAGES: :[[@LINE-1]]:19: warning: {{.*}} in variable declaration with initializer
// CHECK-FIXES: {{^}}void (gronk::*p2)() = &gronk::foo;{{$}}

typedef void (gronk::*member_function_ptr)(void);
// CHECK-MESSAGES: :[[@LINE-1]]:44: warning: {{.*}} in typedef
// CHECK-FIXES: {{^}}typedef void (gronk::*member_function_ptr)();{{$}}

// intentionally not LLVM style to check preservation of whitespace
typedef void (gronk::*member_function_ptr2)
    (
        void
    );
// CHECK-MESSAGES: :[[@LINE-2]]:9: warning: {{.*}} in typedef
// CHECK-FIXES:      {{^typedef void \(gronk::\*member_function_ptr2\)$}}
// CHECK-FIXES-NEXT: {{^    \($}}
// CHECK-FIXES-NEXT: {{^        $}}
// CHECK-FIXES-NEXT: {{^    \);$}}

void gronk::foo() {
  void (*f1)(void) = &::bar;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: {{.*}} in variable declaration with initializer
  // CHECK-FIXES: {{^  }}void (*f1)() = &::bar;{{$}}

  void (*f2)(void);
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: {{.*}} in variable declaration
  // CHECK-FIXES: {{^  }}void (*f2)();{{$}}

  // intentionally not LLVM style to check preservation of whitespace
  void (*f3)
      (
          void
      );
  // CHECK-MESSAGES: :[[@LINE-2]]:11: warning: {{.*}} in variable declaration
  // CHECK-FIXES:      {{^  }}void (*f3){{$}}
  // CHECK-FIXES-NEXT: {{^      \($}}
  // CHECK-FIXES-NEXT: {{^          $}}
  // CHECK-FIXES-NEXT: {{^      \);$}}
}

void gronk::bar(void) {
// CHECK-MESSAGES: :[[@LINE-1]]:17: warning: {{.*}} in function definition
// CHECK-FIXES: {{^}}void gronk::bar() {{{$}}
  void (gronk::*p3)(void) = &gronk::foo;
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: {{.*}} in variable declaration with initializer
  // CHECK-FIXES: {{^  }}void (gronk::*p3)() = &gronk::foo;{{$}}

  void (gronk::*p4)(void);
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: {{.*}} in variable declaration
  // CHECK-FIXES: {{^  }}void (gronk::*p4)();{{$}}

  // intentionally not LLVM style to check preservation of whitespace
  void (gronk::*p5)
      (
          void
      );
  // CHECK-MESSAGES: :[[@LINE-2]]:11: warning: {{.*}} in variable declaration
  // CHECK-FIXES:      {{^  }}void (gronk::*p5){{$}}
  // CHECK-FIXES-NEXT: {{^      \($}}
  // CHECK-FIXES-NExT: {{^          $}}
  // CHECK-FIXES-NExT: {{^      \);$}}
}

// intentionally not LLVM style to check preservation of whitespace
void gronk::bar2
  (
  void
  )
// CHECK-MESSAGES: :[[@LINE-2]]:3: warning: {{.*}} in function definition
// CHECK-FIXES:      {{^void gronk::bar2$}}
// CHECK-FIXES-NEXT: {{^  \($}}
// CHECK-FIXES-NEXT: {{^  $}}
// CHECK-FIXES-NEXT: {{^  \)$}}
{
}

gronk::gronk(void)
// CHECK-MESSAGES: :[[@LINE-1]]:14: warning: {{.*}} in function definition
// CHECK-FIXES: {{^}}gronk::gronk(){{$}}
  : f1(nullptr),
  p1(nullptr) {
}

gronk::~gronk(void) {
// CHECK-MESSAGES: :[[@LINE-1]]:15: warning: {{.*}} in function definition
// CHECK-FIXES: {{^}}gronk::~gronk() {{{$}}
}

class nutter {
public:
  nutter();
};

nutter::nutter(void) {
// CHECK-MESSAGES: :[[@LINE-1]]:16: warning: {{.*}} in function definition
// CHECK-FIXES: {{^}}nutter::nutter() {{{$}}
  void (*f3)(void) = static_cast<void (*)(void)>(0);
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: {{.*}} in variable declaration with initializer
  // CHECK-MESSAGES: :[[@LINE-2]]:43: warning: {{.*}} in named cast
  // CHECK-FIXES: void (*f3)() = static_cast<void (*)()>(0);{{$}}

  void (*f4)(void) = (void (*)(void)) 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: {{.*}} in variable declaration with initializer
  // CHECK-MESSAGES: :[[@LINE-2]]:32: warning: {{.*}} in cast expression
  // CHECK-FIXES: void (*f4)() = (void (*)()) 0;{{$}}

  void (*f5)(void) = reinterpret_cast<void (*)(void)>(0);
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: {{.*}} in variable declaration with initializer
  // CHECK-MESSAGES: :[[@LINE-2]]:48: warning: {{.*}} in named cast
  // CHECK-FIXES: void (*f5)() = reinterpret_cast<void (*)()>(0);{{$}}

  // intentionally not LLVM style to check preservation of whitespace
  void (*f6)(void) = static_cast<void (*)
      (
          void
      )>(0);
  // CHECK-MESSAGES: :[[@LINE-4]]:14: warning: {{.*}} in variable declaration with initializer
  // CHECK-MESSAGES: :[[@LINE-3]]:11: warning: {{.*}} in named cast
  // CHECK-FIXES:      {{^  }}void (*f6)() = static_cast<void (*){{$}}
  // CHECK-FIXES-NEXT: {{^      \($}}
  // CHECK-FIXES-NEXT: {{^          $}}
  // CHECK-FIXES-NEXT: {{^      }})>(0);{{$}}

  // intentionally not LLVM style to check preservation of whitespace
  void (*f7)(void) = (void (*)
      (
          void
      )) 0;
  // CHECK-MESSAGES: :[[@LINE-4]]:14: warning: {{.*}} in variable declaration with initializer
  // CHECK-MESSAGES: :[[@LINE-3]]:11: warning: {{.*}} in cast expression
  // CHECK-FIXES:      {{^  }}void (*f7)() = (void (*){{$}}
  // CHECK-FIXES-NEXT: {{^      \($}}
  // CHECK-FIXES-NEXT: {{^          $}}
  // CHECK-FIXES-NEXT: {{^      \)\) 0;$}}

  // intentionally not LLVM style to check preservation of whitespace
  void (*f8)(void) = reinterpret_cast<void (*)
      (
          void
      )>(0);
  // CHECK-MESSAGES: :[[@LINE-4]]:14: warning: {{.*}} in variable declaration with initializer
  // CHECK-MESSAGES: :[[@LINE-3]]:11: warning: {{.*}} in named cast
  // CHECK-FIXES:      {{^  }}void (*f8)() = reinterpret_cast<void (*){{$}}
  // CHECK-FIXES-NEXT: {{^      \($}}
  // CHECK-FIXES-NEXT: {{^          $}}
  // CHECK-FIXES-NEXT: {{^      \)>\(0\);$}}

  void (*o1)(int) = static_cast<void (*)(int)>(0);
  void (*o2)(int) = (void (*)(int)) 0;
  void (*o3)(int) = reinterpret_cast<void (*)(int)>(0);
}

class generator {
public:
  int operator()(void) { return 1; }
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: {{.*}} in function definition
  // CHECK-FIXES: {{^  }}int operator()() { return 1; }{{$}}
};

void test_lambda_functions() {
  auto lamb_duh = [](void (*fn)(void)) { (*fn)(); };
  // CHECK-MESSAGES: :[[@LINE-1]]:33: warning: {{.*}} in variable declaration
  // CHECK-FIXES: {{^  }}auto lamb_duh = [](void (*fn)()) { (*fn)(); };{{$}}

  auto lambda_generator = [](void) { return 1; };
  // CHECK-MESSAGES: :[[@LINE-1]]:30: warning: {{.*}} in lambda expression
  // CHECK-FIXES: {{^  }}auto lambda_generator = []() { return 1; };{{$}}

  auto gen2 = []() { return 1; };

  auto gen3 = []{ return 1; };

  auto void_returner = [](void) -> void (*)(void) { return f1; };
  // CHECK-MESSAGES: [[@LINE-1]]:27: warning: {{.*}} in lambda expression
  // CHECK-MESSAGES: [[@LINE-2]]:45: warning: {{.*}} in lambda expression
  // CHECK-FIXES: {{^  }}auto void_returner = []() -> void (*)() { return f1; };{{$}}
}

#define M(x) x

M(void inmacro(void) {})
// CHECK-MESSAGES: :[[@LINE-1]]:16: warning: {{.*}} in function definition
// CHECK-FIXES: M(void inmacro() {})

#define F(A, B)        \
  struct F_##A##_##B { \
    F_##A##_##B(void); \
  };                   \
  F_##A##_##B::F_##A##_##B(void)

F(Foo, Bar) {

}

struct DefinitionWithNoBody {
  DefinitionWithNoBody(void) = delete;
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: {{.*}} in function definition
  // CHECK-FIXES: DefinitionWithNoBody() = delete;
};



#define BODY {}
#define LAMBDA1 [](void){}
// CHECK-MESSAGES: :[[@LINE-1]]:20: warning: redundant void argument list in lambda expression [modernize-redundant-void-arg]
// CHECK-FIXES: LAMBDA1 [](){}

#define LAMBDA2 [](void)BODY
// CHECK-MESSAGES: :[[@LINE-1]]:20: warning: redundant void argument list in lambda expression [modernize-redundant-void-arg]
// CHECK-FIXES: LAMBDA2 []()BODY

#define LAMBDA3(captures, args, body) captures args body
#define WRAP(...) __VA_ARGS__

#define LAMBDA4 (void)LAMBDA3([],(void),BODY)
// CHECK-MESSAGES: :[[@LINE-1]]:35: warning: redundant void argument list in lambda expression [modernize-redundant-void-arg]
// CHECK-FIXES: LAMBDA4 (void)LAMBDA3([],(),BODY)

#define LAMBDA5 []() -> void (*)(void) {return BODY;}
// CHECK-MESSAGES: :[[@LINE-1]]:34: warning: redundant void argument list in lambda expression [modernize-redundant-void-arg]
// CHECK-FIXES: LAMBDA5 []() -> void (*)() {return BODY;}
void lambda_expression_with_macro_test(){
  (void)LAMBDA1;
  (void)LAMBDA2;
  (void)LAMBDA3([], (void), BODY);
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: redundant void argument list in lambda expression [modernize-redundant-void-arg]
  // CHECK-FIXES: (void)LAMBDA3([], (), BODY);

  LAMBDA4;
  LAMBDA5;
  WRAP((void)WRAP(WRAP(LAMBDA3(WRAP([]), WRAP((void)), WRAP(BODY)))));
  // CHECK-MESSAGES: :[[@LINE-1]]:48: warning: redundant void argument list in lambda expression [modernize-redundant-void-arg]
  // CHECK-FIXES: WRAP((void)WRAP(WRAP(LAMBDA3(WRAP([]), WRAP(()), WRAP(BODY)))));

  (void)WRAP([](void) {});
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: redundant void argument list in lambda expression [modernize-redundant-void-arg]
  // CHECK-FIXES: (void)WRAP([]() {});

  [](void) BODY;
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: redundant void argument list in lambda expression [modernize-redundant-void-arg]
  // CHECK-FIXES: []() BODY;
}

namespace qqq {
void foo() BODY
void bar(void) BODY;
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: redundant void argument list in function definition
// CHECK-FIXES: void bar() BODY;
}

struct S_1 {
  void g_1(void) const {
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: redundant void argument list in function definition [modernize-redundant-void-arg]
    // CHECK-FIXES: void g_1() const {
    int a;
    (void)a;
  }

  void g_2() const {
    int a;
    (void)a;
  }
};

template <typename T0>
struct S_2 {
  void g_1(void) const {
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: redundant void argument list in function definition [modernize-redundant-void-arg]
    // CHECK-FIXES: void g_1() const {
    int a;
    (void)a;
  }

  void g_2() const {
    int a;
    (void)a;
  }
};

template <typename T0>
struct S_3 {
  template <typename T1>
  void g_1(void) const {
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: redundant void argument list in function definition [modernize-redundant-void-arg]
    // CHECK-FIXES: void g_1() const {
    int a;
    (void)a;
  }
  template <typename T2>
  void g_2() const {
    int a;
    (void)a;
  }
};

template <typename T1>
void g_3(void) {
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: redundant void argument list in function definition [modernize-redundant-void-arg]
  // CHECK-FIXES: void g_3() {
  int a;
  (void)a;
}

//Template instantiation
void f_testTemplate() {
  S_1();
  S_2<int>();
  S_3<int>();
  g_3<int>();
}
