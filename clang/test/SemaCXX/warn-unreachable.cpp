// RUN: %clang_cc1 %s -fcxx-exceptions -fexceptions -fsyntax-only -verify -fblocks -std=c++11 -Wunreachable-code-aggressive -Wno-unused-value -Wno-tautological-compare

int &halt() __attribute__((noreturn));
int &live();
int dead();
int liveti() throw(int);
int (*livetip)() throw(int);

int test1() {
  try {
    live();
  } catch (int i) {
    live();
  }
  return 1;
}

void test2() {
  try {
    live();
  } catch (int i) {
    live();
  }
  try {
    liveti();
  } catch (int i) {
    live();
  }
  try {
    livetip();
  } catch (int i) {
    live();
  }
  throw 1;
  dead();       // expected-warning {{will never be executed}}
}


void test3() {
  halt()
    --;         // expected-warning {{will never be executed}}
  // FIXME: The unreachable part is just the '?', but really all of this
  // code is unreachable and shouldn't be separately reported.
  halt()        // expected-warning {{will never be executed}}
    ? 
    dead() : dead();
  live(),
    float       
      (halt()); // expected-warning {{will never be executed}}
}

namespace Test4 {
  struct S {
    int mem;
  } s;
  S &foor();
  void test4() {
    halt(), foor()// expected-warning {{will never be executed}}
      .mem;       
  }
}

namespace Test5 {
  struct S {
    int mem;
  } s;
  S &foonr() __attribute__((noreturn));
  void test5() {
    foonr()
      .mem;       // expected-warning {{will never be executed}}
  }
}

void test6() {
  struct S {
    ~S() { }
    S(int i) { }
  };
  live(),
    S
      (halt());  // expected-warning {{will never be executed}}
}

// Don't warn about unreachable code in template instantiations, as
// they may only be unreachable in that specific instantiation.
void isUnreachable();

template <typename T> void test_unreachable_templates() {
  T::foo();
  isUnreachable();  // no-warning
}

struct TestUnreachableA {
  static void foo() __attribute__((noreturn));
};
struct TestUnreachableB {
  static void foo();
};

void test_unreachable_templates_harness() {
  test_unreachable_templates<TestUnreachableA>();
  test_unreachable_templates<TestUnreachableB>(); 
}

// Do warn about explicit template specializations, as they represent
// actual concrete functions that somebody wrote.

template <typename T> void funcToSpecialize() {}
template <> void funcToSpecialize<int>() {
  halt();
  dead(); // expected-warning {{will never be executed}}
}

// Handle 'try' code dominating a dead return.
enum PR19040_test_return_t
{ PR19040_TEST_FAILURE };
namespace PR19040_libtest
{
  class A {
  public:
    ~A ();
  };
}
PR19040_test_return_t PR19040_fn1 ()
{
    try
    {
        throw PR19040_libtest::A ();
    } catch (...)
    {
        return PR19040_TEST_FAILURE;
    }
    return PR19040_TEST_FAILURE; // expected-warning {{will never be executed}}
}

__attribute__((noreturn))
void raze();

namespace std {
template<typename T> struct basic_string {
  basic_string(const T* x) {}
  ~basic_string() {};
};
typedef basic_string<char> string;
}

std::string testStr() {
  raze();
  return ""; // expected-warning {{'return' will never be executed}}
}

std::string testStrWarn(const char *s) {
  raze();
  return s; // expected-warning {{will never be executed}}
}

bool testBool() {
  raze();
  return true; // expected-warning {{'return' will never be executed}}
}

static const bool ConditionVar = 1;
int test_global_as_conditionVariable() {
  if (ConditionVar)
    return 1;
  return 0; // no-warning
}

// Handle unreachable temporary destructors.
class A {
public:
  A();
  ~A();
};

__attribute__((noreturn))
void raze(const A& x);

void test_with_unreachable_tmp_dtors(int x) {
  raze(x ? A() : A()); // no-warning
}

// Test sizeof - sizeof in enum declaration.
enum { BrownCow = sizeof(long) - sizeof(char) };
enum { CowBrown = 8 - 1 };


int test_enum_sizeof_arithmetic() {
  if (BrownCow)
    return 1;
  return 2;
}

int test_enum_arithmetic() {
  if (CowBrown)
    return 1;
  return 2; // expected-warning {{never be executed}}
}

int test_arithmetic() {
  if (8 -1)
    return 1;
  return 2; // expected-warning {{never be executed}}
}

int test_treat_const_bool_local_as_config_value() {
  const bool controlValue = false;
  if (!controlValue)
    return 1;
  test_treat_const_bool_local_as_config_value(); // no-warning
  return 0;
}

int test_treat_non_const_bool_local_as_non_config_value() {
  bool controlValue = false;
  if (!controlValue)
    return 1;
  // There is no warning here because 'controlValue' isn't really
  // a control value at all.  The CFG will not treat this
  // branch as unreachable.
  test_treat_non_const_bool_local_as_non_config_value(); // no-warning
  return 0;
}

void test_do_while(int x) {
  // Handle trivial expressions with
  // implicit casts to bool.
  do {
    break;
  } while (0); // no-warning
}

class Frobozz {
public:
  Frobozz(int x);
  ~Frobozz();
};

Frobozz test_return_object(int flag) {
  return Frobozz(flag);
  return Frobozz(42);  // expected-warning {{'return' will never be executed}}
}

Frobozz test_return_object_control_flow(int flag) {
  return Frobozz(flag);
  return Frobozz(flag ? 42 : 24); // expected-warning {{code will never be executed}}
}

void somethingToCall();

static constexpr bool isConstExprConfigValue() { return true; }

int test_const_expr_config_value() {
 if (isConstExprConfigValue()) {
   somethingToCall();
   return 0;
 }
 somethingToCall(); // no-warning
 return 1;
}
int test_const_expr_config_value_2() {
 if (!isConstExprConfigValue()) {
   somethingToCall(); // no-warning
   return 0;
 }
 somethingToCall();
 return 1;
}

class Frodo {
public:
  static const bool aHobbit = true;
};

void test_static_class_var() {
  if (Frodo::aHobbit)
    somethingToCall();
  else
    somethingToCall(); // no-warning
}

void test_static_class_var(Frodo &F) {
  if (F.aHobbit)
    somethingToCall();
  else
    somethingToCall(); // no-warning
}

void test_unreachable_for_null_increment() {
  for (unsigned i = 0; i < 10 ; ) // no-warning
    break;
}

void test_unreachable_forrange_increment() {
  int x[10] = { 0 };
  for (auto i : x) { // expected-warning {{loop will run at most once (loop increment never executed)}}
    break;
  }
}

void calledFun() {}

// Test "silencing" with parentheses.
void test_with_paren_silencing(int x) {
  if (false) calledFun(); // expected-warning {{will never be executed}} expected-note {{silence by adding parentheses to mark code as explicitly dead}}
  if ((false)) calledFun(); // no-warning

  if (true) // expected-note {{silence by adding parentheses to mark code as explicitly dead}}
    calledFun();
  else
    calledFun(); // expected-warning {{will never be executed}}

  if ((true))
    calledFun();
  else
    calledFun(); // no-warning
  
  if (!true) // expected-note {{silence by adding parentheses to mark code as explicitly dead}}
    calledFun(); // expected-warning {{code will never be executed}}
  else
    calledFun();
  
  if ((!true))
    calledFun(); // no-warning
  else
    calledFun();
  
  if (!(true))
    calledFun(); // no-warning
  else
    calledFun();
}

void test_with_paren_silencing_impcast(int x) {
  if (0) calledFun(); // expected-warning {{will never be executed}} expected-note {{silence by adding parentheses to mark code as explicitly dead}}
  if ((0)) calledFun(); // no-warning

  if (1) // expected-note {{silence by adding parentheses to mark code as explicitly dead}}
    calledFun();
  else
    calledFun(); // expected-warning {{will never be executed}}

  if ((1))
    calledFun();
  else
    calledFun(); // no-warning
  
  if (!1) // expected-note {{silence by adding parentheses to mark code as explicitly dead}}
    calledFun(); // expected-warning {{code will never be executed}}
  else
    calledFun();
  
  if ((!1))
    calledFun(); // no-warning
  else
    calledFun();
  
  if (!(1))
    calledFun(); // no-warning
  else
    calledFun();
}

void tautological_compare(bool x, int y) {
  if (x > 10)           // expected-note {{silence}}
    calledFun();        // expected-warning {{will never be executed}}
  if (10 < x)           // expected-note {{silence}}
    calledFun();        // expected-warning {{will never be executed}}
  if (x == 10)          // expected-note {{silence}}
    calledFun();        // expected-warning {{will never be executed}}

  if (x < 10)           // expected-note {{silence}}
    calledFun();
  else
    calledFun();        // expected-warning {{will never be executed}}
  if (10 > x)           // expected-note {{silence}}
    calledFun();
  else
    calledFun();        // expected-warning {{will never be executed}}
  if (x != 10)          // expected-note {{silence}}
    calledFun();
  else
    calledFun();        // expected-warning {{will never be executed}}

  if (y != 5 && y == 5) // expected-note {{silence}}
    calledFun();        // expected-warning {{will never be executed}}

  if (y > 5 && y < 4)   // expected-note {{silence}}
    calledFun();        // expected-warning {{will never be executed}}

  if (y < 10 || y > 5)  // expected-note {{silence}}
    calledFun();
  else
    calledFun();        // expected-warning {{will never be executed}}

  if (y == -1 && y != -1)  // expected-note {{silence}}
    calledFun();        // expected-warning {{will never be executed}}

  if (x == -1)   // expected-note {{silence}}
    calledFun(); // expected-warning {{will never be executed}}

  if (x != -1)   // expected-note {{silence}}
    calledFun();
  else
    calledFun(); // expected-warning {{will never be executed}}

  // TODO: Extend warning to the following code:
  if (x < -1)
    calledFun();

  if (-1 > x)
    calledFun();
  else
    calledFun();

}
