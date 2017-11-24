// RUN: %check_clang_tidy %s bugprone-forward-declaration-namespace %t

namespace {
// This is a declaration in a wrong namespace.
class T_A;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: declaration 'T_A' is never referenced, but a declaration with the same name found in another namespace 'na' [bugprone-forward-declaration-namespace]
// CHECK-MESSAGES: note: a declaration of 'T_A' is found here
// CHECK-MESSAGES: :[[@LINE-3]]:7: warning: no definition found for 'T_A', but a definition with the same name 'T_A' found in another namespace '(global)' [bugprone-forward-declaration-namespace]
// CHECK-MESSAGES: note: a definition of 'T_A' is found here
}

namespace na {
// This is a declaration in a wrong namespace.
class T_A;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: declaration 'T_A' is never referenced, but a declaration with the same name found in another namespace '(anonymous)'
// CHECK-MESSAGES: note: a declaration of 'T_A' is found here
// CHECK-MESSAGES: :[[@LINE-3]]:7: warning: no definition found for 'T_A', but a definition with the same name 'T_A' found in another namespace '(global)'
// CHECK-MESSAGES: note: a definition of 'T_A' is found here
}

class T_A;

class T_A {
  int x;
};

class NESTED;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: no definition found for 'NESTED', but a definition with the same name 'NESTED' found in another namespace '(anonymous namespace)::nq::(anonymous)'
// CHECK-MESSAGES: note: a definition of 'NESTED' is found here

namespace {
namespace nq {
namespace {
class NESTED {};
}
}
}

namespace na {
class T_B;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: declaration 'T_B' is never referenced, but a declaration with the same name found in another namespace 'nb'
// CHECK-MESSAGES: note: a declaration of 'T_B' is found here
// CHECK-MESSAGES: :[[@LINE-3]]:7: warning: no definition found for 'T_B', but a definition with the same name 'T_B' found in another namespace 'nb'
// CHECK-MESSAGES: note: a definition of 'T_B' is found here
}

namespace nb {
class T_B;
}

namespace nb {
class T_B {
  int x;
};
}

namespace na {
class T_B;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: declaration 'T_B' is never referenced, but a declaration with the same name found in another namespace 'nb'
// CHECK-MESSAGES: note: a declaration of 'T_B' is found here
// CHECK-MESSAGES: :[[@LINE-3]]:7: warning: no definition found for 'T_B', but a definition with the same name 'T_B' found in another namespace 'nb'
// CHECK-MESSAGES: note: a definition of 'T_B' is found here
}

// A simple forward declaration. Although it is never used, but no declaration
// with the same name is found in other namespace.
class OUTSIDER;

namespace na {
// This class is referenced declaration, we don't generate warning.
class OUTSIDER_1;
}

void f(na::OUTSIDER_1);

namespace nc {
// This class is referenced as friend in OOP.
class OUTSIDER_1;

class OOP {
  friend struct OUTSIDER_1;
};
}

namespace nd {
class OUTSIDER_1;
void f(OUTSIDER_1 *);
}

namespace nb {
class OUTSIDER_1;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: declaration 'OUTSIDER_1' is never referenced, but a declaration with the same name found in another namespace 'na'
// CHECK-MESSAGES: note: a declaration of 'OUTSIDER_1' is found here
}


namespace na {
template<typename T>
class T_C;
}

namespace nb {
// FIXME: this is an error, but we don't consider template class declaration
// now.
template<typename T>
class T_C;
}

namespace na {
template<typename T>
class T_C {
  int x;
};
}

namespace na {

template <typename T>
class T_TEMP {
  template <typename _Tp1>
  struct rebind { typedef T_TEMP<_Tp1> other; };
};

// We ignore class template specialization.
template class T_TEMP<char>;
}

namespace nb {

template <typename T>
class T_TEMP_1 {
  template <typename _Tp1>
  struct rebind { typedef T_TEMP_1<_Tp1> other; };
};

// We ignore class template specialization.
extern template class T_TEMP_1<char>;
}

namespace nd {
class D;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: declaration 'D' is never referenced, but a declaration with the same name found in another namespace 'nd::ne'
// CHECK-MESSAGES: note: a declaration of 'D' is found here
}

namespace nd {
namespace ne {
class D;
}
}

int f(nd::ne::D &d);


// This should be ignored by the check.
template <typename... Args>
class Observer {
  class Impl;
};

template <typename... Args>
class Observer<Args...>::Impl {
};
