// RUN: rm -rf %t && mkdir %t
// RUN: split-file %s %t


// RUN: %check_clang_tidy -std=c++11,c++14,c++17,c++20 -check-suffix=DEFAULT %t/cppcoreguidelines-pro-bounds-avoid-unchecked-container-access.cpp \
// RUN: cppcoreguidelines-pro-bounds-avoid-unchecked-container-access %t -- -- -I%t

// RUN: %check_clang_tidy -std=c++11,c++14,c++17,c++20 -check-suffix=AT %t/cppcoreguidelines-pro-bounds-avoid-unchecked-container-access.cpp \
// RUN: cppcoreguidelines-pro-bounds-avoid-unchecked-container-access %t -- \
// RUN: -config='{CheckOptions: {cppcoreguidelines-pro-bounds-avoid-unchecked-container-access.FixMode: at}}' -- -I%t

// RUN: %check_clang_tidy -std=c++11,c++14,c++17,c++20 -check-suffix=FUNC %t/cppcoreguidelines-pro-bounds-avoid-unchecked-container-access.cpp \
// RUN: cppcoreguidelines-pro-bounds-avoid-unchecked-container-access %t -- \
// RUN: -config='{CheckOptions: {cppcoreguidelines-pro-bounds-avoid-unchecked-container-access.FixMode: function, \
// RUN: cppcoreguidelines-pro-bounds-avoid-unchecked-container-access.FixFunction: "f"}}' -- -I%t


// RUN: %check_clang_tidy -std=c++11,c++14,c++17,c++20 -check-suffix=DEFAULT-NO-EXCL %t/cppcoreguidelines-pro-bounds-avoid-unchecked-container-access-no-excl.cpp \
// RUN: cppcoreguidelines-pro-bounds-avoid-unchecked-container-access %t -- \
// RUN: -config='{CheckOptions: {cppcoreguidelines-pro-bounds-avoid-unchecked-container-access.ExcludeClasses: ""}}' -- -I%t

// RUN: %check_clang_tidy -std=c++11,c++14,c++17,c++20 -check-suffix=AT-NO-EXCL %t/cppcoreguidelines-pro-bounds-avoid-unchecked-container-access-no-excl.cpp \
// RUN: cppcoreguidelines-pro-bounds-avoid-unchecked-container-access %t -- \
// RUN: -config='{CheckOptions: {cppcoreguidelines-pro-bounds-avoid-unchecked-container-access.ExcludeClasses: "", \
// RUN: cppcoreguidelines-pro-bounds-avoid-unchecked-container-access.FixMode: at}}' -- -I%t

// RUN: %check_clang_tidy -std=c++11,c++14,c++17,c++20 -check-suffix=FUNC-NO-EXCL %t/cppcoreguidelines-pro-bounds-avoid-unchecked-container-access-no-excl.cpp \
// RUN: cppcoreguidelines-pro-bounds-avoid-unchecked-container-access %t -- \
// RUN: -config='{CheckOptions: {cppcoreguidelines-pro-bounds-avoid-unchecked-container-access.ExcludeClasses: "", \
// RUN: cppcoreguidelines-pro-bounds-avoid-unchecked-container-access.FixMode: function, \
// RUN: cppcoreguidelines-pro-bounds-avoid-unchecked-container-access.FixFunction: "f"}}' -- -I%t


// RUN: %check_clang_tidy -std=c++11,c++14,c++17,c++20 -check-suffix=DEFAULT-EXCL %t/cppcoreguidelines-pro-bounds-avoid-unchecked-container-access-excl.cpp \
// RUN: cppcoreguidelines-pro-bounds-avoid-unchecked-container-access %t -- \
// RUN: -config='{CheckOptions: {cppcoreguidelines-pro-bounds-avoid-unchecked-container-access.ExcludeClasses: "ExcludedClass1;ExcludedClass2"}}' -- -I%t

// RUN: %check_clang_tidy -std=c++11,c++14,c++17,c++20 -check-suffix=AT-EXCL %t/cppcoreguidelines-pro-bounds-avoid-unchecked-container-access-excl.cpp \
// RUN: cppcoreguidelines-pro-bounds-avoid-unchecked-container-access %t -- \
// RUN: -config='{CheckOptions: {cppcoreguidelines-pro-bounds-avoid-unchecked-container-access.ExcludeClasses: "ExcludedClass1;ExcludedClass2", \
// RUN: cppcoreguidelines-pro-bounds-avoid-unchecked-container-access.FixMode: at}}' -- -I%t

// RUN: %check_clang_tidy -std=c++11,c++14,c++17,c++20 -check-suffix=FUNC-EXCL %t/cppcoreguidelines-pro-bounds-avoid-unchecked-container-access-excl.cpp \
// RUN: cppcoreguidelines-pro-bounds-avoid-unchecked-container-access %t -- \
// RUN: -config='{CheckOptions: {cppcoreguidelines-pro-bounds-avoid-unchecked-container-access.ExcludeClasses: "ExcludedClass1;ExcludedClass2", \
// RUN: cppcoreguidelines-pro-bounds-avoid-unchecked-container-access.FixMode: function, \
// RUN: cppcoreguidelines-pro-bounds-avoid-unchecked-container-access.FixFunction: "f"}}' -- -I%t


// RUN: %check_clang_tidy -std=c++23 -check-suffixes=DEFAULT-CXX-23 %t/cppcoreguidelines-pro-bounds-avoid-unchecked-container-access-cxx-23.cpp \
// RUN: cppcoreguidelines-pro-bounds-avoid-unchecked-container-access %t -- -- -I%t -DCXX_23=1

// RUN: %check_clang_tidy -std=c++23 -check-suffixes=AT-CXX-23 %t/cppcoreguidelines-pro-bounds-avoid-unchecked-container-access-cxx-23.cpp \
// RUN: cppcoreguidelines-pro-bounds-avoid-unchecked-container-access %t -- \
// RUN: -config='{CheckOptions: {cppcoreguidelines-pro-bounds-avoid-unchecked-container-access.FixMode: at}}' -- -I%t -DCXX_23=1 

// RUN: %check_clang_tidy -std=c++23 -check-suffixes=FUNC-CXX-23 %t/cppcoreguidelines-pro-bounds-avoid-unchecked-container-access-cxx-23.cpp \
// RUN: cppcoreguidelines-pro-bounds-avoid-unchecked-container-access %t -- \
// RUN: -config='{CheckOptions: {cppcoreguidelines-pro-bounds-avoid-unchecked-container-access.FixMode: function, \
// RUN: cppcoreguidelines-pro-bounds-avoid-unchecked-container-access.FixFunction: "f"}}' -- -I%t -DCXX_23=1 

// RUN: %check_clang_tidy -std=c++23 -check-suffixes=FUNC-EMPTY-ARGS-CXX-23 %t/cppcoreguidelines-pro-bounds-avoid-unchecked-container-access-cxx-23.cpp \
// RUN: cppcoreguidelines-pro-bounds-avoid-unchecked-container-access %t -- \
// RUN: -config='{CheckOptions: {cppcoreguidelines-pro-bounds-avoid-unchecked-container-access.FixMode: function, \
// RUN: cppcoreguidelines-pro-bounds-avoid-unchecked-container-access.FixFunction: "f", cppcoreguidelines-pro-bounds-avoid-unchecked-container-access.FixFunctionEmptyArgs: "g", }}' -- -I%t -DCXX_23=1

// RUN: %check_clang_tidy -std=c++23 -check-suffix=FUNC-EMPTY-ARGS-EMPTY-CXX-23 %t/cppcoreguidelines-pro-bounds-avoid-unchecked-container-access-cxx-23.cpp \
// RUN: cppcoreguidelines-pro-bounds-avoid-unchecked-container-access %t -- \
// RUN: -config='{CheckOptions: {cppcoreguidelines-pro-bounds-avoid-unchecked-container-access.FixMode: function, \
// RUN: cppcoreguidelines-pro-bounds-avoid-unchecked-container-access.FixFunctionEmptyArgs: "", }}' -- -I%t -DCXX_23=1

//--- cppcoreguidelines-pro-bounds-avoid-unchecked-container-access.h

namespace std {
  template<typename T, unsigned size>
  struct array {
    T operator[](unsigned i) {
      return T{1};
    }
    T at(unsigned i) {
      return T{1};
    }
    T at() {
      return T{1};
    }
  };

  template<typename T, typename V>
  struct map {
    T operator[](unsigned i) {
      return T{1};
    }
    T at(unsigned i) {
      return T{1};
    }
  };

  template<typename T>
  struct unique_ptr {
    T operator[](unsigned i) {
      return T{1};
    }
  };

  template<typename T>
  struct span {
    T operator[](unsigned i) {
      return T{1};
    }
  };
} // namespace std

namespace json {
  template<typename T>
  struct node{
    T operator[](unsigned i) {
      return T{1};
    }
  };
} // namespace json

//--- cppcoreguidelines-pro-bounds-avoid-unchecked-container-access.cpp

#include "cppcoreguidelines-pro-bounds-avoid-unchecked-container-access.h"

struct SubClass : std::array<int, 3> {};

template<class T> int f(T, unsigned){ return 0;}
template<class T> int f(T){ return 0;}

std::array<int, 3> a;

auto b = a[0];
// CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:11: warning: possibly unsafe 'operator[]', consider bounds-safe alternatives [cppcoreguidelines-pro-bounds-avoid-unchecked-container-access]
// CHECK-FIXES-AT: auto b = a.at(0);
// CHECK-FIXES-FUNC: auto b = f(a, 0);

auto c = a[1+1];
// CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:11: warning: possibly unsafe 'operator[]', consider bounds-safe alternatives [cppcoreguidelines-pro-bounds-avoid-unchecked-container-access]
// CHECK-FIXES-AT: auto c = a.at(1+1);
// CHECK-FIXES-FUNC: auto c = f(a, 1+1);

constexpr int Index = 1;

auto d = a[Index];
// CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:11: warning: possibly unsafe 'operator[]', consider bounds-safe alternatives [cppcoreguidelines-pro-bounds-avoid-unchecked-container-access]
// CHECK-FIXES-AT: auto d = a.at(Index);
// CHECK-FIXES-FUNC: auto d = f(a, Index);

int e(int Ind) {
  return a[Ind];
  // CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:11: warning: possibly unsafe 'operator[]', consider bounds-safe alternatives [cppcoreguidelines-pro-bounds-avoid-unchecked-container-access]
  // CHECK-FIXES-AT: return a.at(Ind);
  // CHECK-FIXES-FUNC: return f(a, Ind);
}

auto fa = (&a)->operator[](1);
// CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:11: warning: possibly unsafe 'operator[]', consider bounds-safe alternatives [cppcoreguidelines-pro-bounds-avoid-unchecked-container-access]
// CHECK-FIXES-AT: auto fa = (&a)->at(1);
// CHECK-FIXES-FUNC: auto fa = f(*(&a), 1);

auto fd = a.operator[](1);
// CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:11: warning: possibly unsafe 'operator[]', consider bounds-safe alternatives [cppcoreguidelines-pro-bounds-avoid-unchecked-container-access]
// CHECK-FIXES-AT: auto fd = a.at(1);
// CHECK-FIXES-FUNC: auto fd = f(a, 1);



auto g = a.at(0);

std::unique_ptr<int> p;
auto q = p[0];
// CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:11: warning: possibly unsafe 'operator[]', consider bounds-safe alternatives [cppcoreguidelines-pro-bounds-avoid-unchecked-container-access]
// CHECK-FIXES-AT: auto q = p[0];
// CHECK-FIXES-FUNC: auto q = f(p, 0);

std::span<int> s;
auto t = s[0];
// CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:11: warning: possibly unsafe 'operator[]', consider bounds-safe alternatives [cppcoreguidelines-pro-bounds-avoid-unchecked-container-access]
// CHECK-FIXES-AT: auto t = s[0];
// CHECK-FIXES-FUNC: auto t = f(s, 0);

json::node<int> n;
auto m = n[0];
// CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:11: warning: possibly unsafe 'operator[]', consider bounds-safe alternatives [cppcoreguidelines-pro-bounds-avoid-unchecked-container-access]
// CHECK-FIXES-AT: auto m = n[0];
// CHECK-FIXES-FUNC: auto m = f(n, 0);

SubClass Sub;
auto r = Sub[0];
// CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:13: warning: possibly unsafe 'operator[]', consider bounds-safe alternatives [cppcoreguidelines-pro-bounds-avoid-unchecked-container-access]
// CHECK-FIXES-AT: auto r = Sub.at(0);
// CHECK-FIXES-FUNC: auto r = f(Sub, 0);

typedef std::array<int, 3> ar;
ar BehindDef;
auto u = BehindDef[0];
// CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:19: warning: possibly unsafe 'operator[]', consider bounds-safe alternatives [cppcoreguidelines-pro-bounds-avoid-unchecked-container-access]
// CHECK-FIXES-AT: auto u = BehindDef.at(0);
// CHECK-FIXES-FUNC: auto u = f(BehindDef, 0);

template<typename T> int TestTemplate(T t){
  return t[0];
  // CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:10: warning: possibly unsafe 'operator[]', consider bounds-safe alternatives [cppcoreguidelines-pro-bounds-avoid-unchecked-container-access]

}


auto v = TestTemplate<>(a);
auto w = TestTemplate<>(p);

#define SUBSCRIPT_BEHIND_MACRO(x) a[x]
#define ARG_BEHIND_MACRO 0
#define OBJECT_BEHIND_MACRO a

auto m1 = SUBSCRIPT_BEHIND_MACRO(0);
// CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:11: warning: possibly unsafe 'operator[]', consider bounds-safe alternatives [cppcoreguidelines-pro-bounds-avoid-unchecked-container-access]

auto m2 = a[ARG_BEHIND_MACRO];
// CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:12: warning: possibly unsafe 'operator[]', consider bounds-safe alternatives [cppcoreguidelines-pro-bounds-avoid-unchecked-container-access]
// CHECK-FIXES-AT: auto m2 = a.at(ARG_BEHIND_MACRO);
// CHECK-FIXES-FUNC: auto m2 = f(a, ARG_BEHIND_MACRO);

auto m3 = OBJECT_BEHIND_MACRO[0];
// CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:30: warning: possibly unsafe 'operator[]', consider bounds-safe alternatives [cppcoreguidelines-pro-bounds-avoid-unchecked-container-access]
// CHECK-FIXES-AT: auto m3 = OBJECT_BEHIND_MACRO.at(0);
// CHECK-FIXES-FUNC: auto m3 = f(OBJECT_BEHIND_MACRO, 0);

// Check that spacing does not invalidate the fixes 
std::array<int , 3> longname;

auto z1 = longname   [    0    ]  ;
// CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:22: warning: possibly unsafe 'operator[]', consider bounds-safe alternatives [cppcoreguidelines-pro-bounds-avoid-unchecked-container-access]
// CHECK-FIXES-AT: auto z1 = longname   .at(    0    )  ;
// CHECK-FIXES-FUNC: auto z1 = f(longname   ,     0    )  ;
auto z2 = longname   . operator[]   ( 0 );
// CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:11: warning: possibly unsafe 'operator[]', consider bounds-safe alternatives [cppcoreguidelines-pro-bounds-avoid-unchecked-container-access]
// CHECK-FIXES-AT: auto z2 = longname   . at   ( 0 );
// CHECK-FIXES-FUNC: auto z2 = f(longname   ,  0 );
auto z3 = (&longname)   -> operator[]   ( 0 );
// CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:11: warning: possibly unsafe 'operator[]', consider bounds-safe alternatives [cppcoreguidelines-pro-bounds-avoid-unchecked-container-access]
// CHECK-FIXES-AT: auto z3 = (&longname)   -> at   ( 0 );
// CHECK-FIXES-FUNC: auto z3 = f(*(&longname)   ,  0 );


//--- cppcoreguidelines-pro-bounds-avoid-unchecked-container-access-no-excl.cpp

#include "cppcoreguidelines-pro-bounds-avoid-unchecked-container-access.h"

class ExcludedClass1 {
  public:
    int operator[](unsigned i) {
      return 1;
    }
    int at(unsigned i) {
      return 1;
    }
};

class ExcludedClass2 {
  public:
    int operator[](unsigned i) {
      return 1;
    }
    int at(unsigned i) {
      return 1;
    }
};

ExcludedClass1 E1;
auto x1 = E1[0];
// CHECK-MESSAGES-DEFAULT-NO-EXCL: :[[@LINE-1]]:13: warning: possibly unsafe 'operator[]', consider bounds-safe alternatives [cppcoreguidelines-pro-bounds-avoid-unchecked-container-access]
// CHECK-FIXES-AT-NO-EXCL: auto x1 = E1.at(0);
// CHECK-FIXES-FUNC-NO-EXCL: auto x1 = f(E1, 0);

ExcludedClass2 E2;
auto x2 = E2[0];
// CHECK-MESSAGES-DEFAULT-NO-EXCL: :[[@LINE-1]]:13: warning: possibly unsafe 'operator[]', consider bounds-safe alternatives [cppcoreguidelines-pro-bounds-avoid-unchecked-container-access]
// CHECK-FIXES-AT-NO-EXCL: auto x2 = E2.at(0);
// CHECK-FIXES-FUNC-NO-EXCL: auto x2 = f(E2, 0);

std::map<int,int> TestMapNoExcl;
auto y = TestMapNoExcl[0];
// CHECK-MESSAGES-DEFAULT-NO-EXCL: :[[@LINE-1]]:23: warning: possibly unsafe 'operator[]', consider bounds-safe alternatives [cppcoreguidelines-pro-bounds-avoid-unchecked-container-access]
// CHECK-FIXES-AT-NO-EXCL: auto y = TestMapNoExcl.at(0);
// CHECK-FIXES-FUNC-NO-EXCL: auto y = f(TestMapNoExcl, 0);


//--- cppcoreguidelines-pro-bounds-avoid-unchecked-container-access-excl.cpp

#include "cppcoreguidelines-pro-bounds-avoid-unchecked-container-access.h"

std::map<int,int> TestMapExcl;
auto y = TestMapExcl[0];
// CHECK-MESSAGES-DEFAULT-EXCL: :[[@LINE-1]]:21: warning: possibly unsafe 'operator[]', consider bounds-safe alternatives [cppcoreguidelines-pro-bounds-avoid-unchecked-container-access]
// CHECK-FIXES-AT-EXCL: auto y = TestMapExcl.at(0);   
// CHECK-FIXES-FUNC-EXCL: auto y = f(TestMapExcl, 0); 


//--- cppcoreguidelines-pro-bounds-avoid-unchecked-container-access-cxx-23.cpp
#ifdef CXX_23
#include "cppcoreguidelines-pro-bounds-avoid-unchecked-container-access.h"

namespace std {
  template<typename T, unsigned size>
  struct array_cxx_23 {
    T operator[]() {
      return T{1};
    }
    T at() {
      return T{1};
    }
  };
};

std::array_cxx_23<int, 3> a;

auto b23 = a[];
// CHECK-MESSAGES-DEFAULT-CXX-23: :[[@LINE-1]]:13: warning: possibly unsafe 'operator[]', consider bounds-safe alternatives [cppcoreguidelines-pro-bounds-avoid-unchecked-container-access]
// CHECK-FIXES-AT-CXX-23: auto b23 = a.at();
// CHECK-FIXES-FUNC-CXX-23: auto b23 = f(a);
// CHECK-FIXES-FUNC-EMPTY-ARGS-CXX-23: auto b23 = g(a);
// CHECK-MESSAGES-FUNC-EMPTY-ARGS-EMPTY-CXX-23: :[[@LINE-5]]:13: warning: possibly unsafe 'operator[]' [cppcoreguidelines-pro-bounds-avoid-unchecked-container-access] 
// CHECK-MESSAGES-FUNC-EMPTY-ARGS-EMPTY-CXX-23-NOT: :[[@LINE-6]]:{{.*}}: note: FIX-IT applied suggested code changes 

auto fa23 = (&a)->operator[]();
// CHECK-MESSAGES-DEFAULT-CXX-23: :[[@LINE-1]]:13: warning: possibly unsafe 'operator[]', consider bounds-safe alternatives [cppcoreguidelines-pro-bounds-avoid-unchecked-container-access]
// CHECK-FIXES-AT-CXX-23: auto fa23 = (&a)->at();
// CHECK-FIXES-FUNC-CXX-23: auto fa23 = f(*(&a));
// CHECK-FIXES-FUNC-EMPTY-ARGS-CXX-23: auto fa23 = g(*(&a));
// CHECK-MESSAGES-FUNC-EMPTY-ARGS-EMPTY-CXX-23: :[[@LINE-5]]:13: warning: possibly unsafe 'operator[]' [cppcoreguidelines-pro-bounds-avoid-unchecked-container-access] 
// CHECK-MESSAGES-FUNC-EMPTY-ARGS-EMPTY-CXX-23-NOT: :[[@LINE-6]]:{{.*}}: note: FIX-IT applied suggested code changes 

auto fd23 = a.operator[]();
// CHECK-MESSAGES-DEFAULT-CXX-23: :[[@LINE-1]]:13: warning: possibly unsafe 'operator[]', consider bounds-safe alternatives [cppcoreguidelines-pro-bounds-avoid-unchecked-container-access]
// CHECK-FIXES-AT-CXX-23: auto fd23 = a.at();
// CHECK-FIXES-FUNC-CXX-23: auto fd23 = f(a);
// CHECK-FIXES-FUNC-EMPTY-ARGS-CXX-23: auto fd23 = g(a);
// CHECK-MESSAGES-FUNC-EMPTY-ARGS-EMPTY-CXX-23: :[[@LINE-5]]:13: warning: possibly unsafe 'operator[]' [cppcoreguidelines-pro-bounds-avoid-unchecked-container-access] 
// CHECK-MESSAGES-FUNC-EMPTY-ARGS-EMPTY-CXX-23-NOT: :[[@LINE-6]]:{{.*}}: note: FIX-IT applied suggested code changes 
#endif
