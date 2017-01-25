// RUN: %check_clang_tidy %s modernize-use-using %t

typedef int Type;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef' [modernize-use-using]
// CHECK-FIXES: using Type = int;

typedef long LL;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef'
// CHECK-FIXES: using LL = long;

typedef int Bla;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef'
// CHECK-FIXES: using Bla = int;

typedef Bla Bla2;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef'
// CHECK-FIXES: using Bla2 = Bla;

typedef void (*type)(int, int);
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef'
// CHECK-FIXES: using type = void (*)(int, int);

typedef void (*type2)();
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef'
// CHECK-FIXES: using type2 = void (*)();

class Class {
  typedef long long Type;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use 'using' instead of 'typedef'
  // CHECK-FIXES: using Type = long long;
};

typedef void (Class::*MyPtrType)(Bla) const;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef'
// CHECK-FIXES: using MyPtrType = void (Class::*)(Bla)[[ATTR:( __attribute__\(\(thiscall\)\))?]] const;

class Iterable {
public:
  class Iterator {};
};

template <typename T>
class Test {
  typedef typename T::iterator Iter;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use 'using' instead of 'typedef'
  // CHECK-FIXES: using Iter = typename T::iterator;
};

using balba = long long;

union A {};

typedef void (A::*PtrType)(int, int) const;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef'
// CHECK-FIXES: using PtrType = void (A::*)(int, int)[[ATTR]] const;

typedef Class some_class;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef'
// CHECK-FIXES: using some_class = Class;

typedef Class Cclass;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef'
// CHECK-FIXES: using Cclass = Class;

typedef Cclass cclass2;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef'
// CHECK-FIXES: using cclass2 = Cclass;

class cclass {};

typedef void (cclass::*MyPtrType3)(Bla);
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef'
// CHECK-FIXES: using MyPtrType3 = void (cclass::*)(Bla)[[ATTR]];

using my_class = int;

typedef Test<my_class *> another;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef'
// CHECK-FIXES: using another = Test<my_class *>;

typedef int* PInt;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef'
// CHECK-FIXES: using PInt = int *;

typedef int bla1, bla2, bla3;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef'
// CHECK-FIXES: typedef int bla1, bla2, bla3;

#define CODE typedef int INT

CODE;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef'
// CHECK-FIXES: #define CODE typedef int INT
// CHECK-FIXES: CODE;

struct Foo;
#define Bar Baz
typedef Foo Bar;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef'
// CHECK-FIXES: #define Bar Baz
// CHECK-FIXES: using Baz = Foo;

#define TYPEDEF typedef
TYPEDEF Foo Bak;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef'
// CHECK-FIXES: #define TYPEDEF typedef
// CHECK-FIXES: TYPEDEF Foo Bak;

#define FOO Foo
typedef FOO Bam;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef'
// CHECK-FIXES: #define FOO Foo
// CHECK-FIXES: using Bam = Foo;

typedef struct Foo Bap;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef'
// CHECK-FIXES: using Bap = struct Foo;

struct Foo typedef Bap2;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef'
// CHECK-FIXES: using Bap2 = struct Foo;

Foo typedef Bap3;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef'
// CHECK-FIXES: using Bap3 = Foo;

typedef struct Unknown Baq;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef'
// CHECK-FIXES: using Baq = struct Unknown;

struct Unknown2 typedef Baw;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef'
// CHECK-FIXES: using Baw = struct Unknown2;

int typedef Bax;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef'
// CHECK-FIXES: using Bax = int;

typedef struct Q1 { int a; } S1;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef'
// CHECK-FIXES: typedef struct Q1 { int a; } S1;
typedef struct { int b; } S2;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef'
// CHECK-FIXES: typedef struct { int b; } S2;
struct Q2 { int c; } typedef S3;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef'
// CHECK-FIXES: struct Q2 { int c; } typedef S3;
struct { int d; } typedef S4;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef'
// CHECK-FIXES: struct { int d; } typedef S4;
