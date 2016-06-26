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
// CHECK-FIXES: using MyPtrType = void (Class::*)(Bla) const;

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
// CHECK-FIXES: using PtrType = void (A::*)(int, int) const;

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
// CHECK-FIXES: using MyPtrType3 = void (cclass::*)(Bla);

using my_class = int;

typedef Test<my_class *> another;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef'
// CHECK-FIXES: using another = Test<my_class *>;

typedef int bla1, bla2, bla3;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef'

#define CODE typedef int INT

CODE;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef'
