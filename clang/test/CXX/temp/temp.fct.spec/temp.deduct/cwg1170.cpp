// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s
// expected-no-diagnostics

#if !__has_feature(cxx_access_control_sfinae)
#  error No support for access control as part of SFINAE?
#endif

typedef char yes_type;
typedef char (&no_type)[2];

template<unsigned N> struct unsigned_c { };

template<typename T>
class has_copy_constructor {
  static T t;

  template<typename U> static yes_type check(unsigned_c<sizeof(U(t))> * = 0);
  template<typename U> static no_type check(...);

public:
  static const bool value = (sizeof(check<T>(0)) == sizeof(yes_type));
};

struct HasCopy { };

struct HasNonConstCopy {
  HasNonConstCopy(HasNonConstCopy&);
};

struct HasDeletedCopy {
  HasDeletedCopy(const HasDeletedCopy&) = delete;
};

struct HasPrivateCopy {
private:
  HasPrivateCopy(const HasPrivateCopy&);
};

int check0[has_copy_constructor<HasCopy>::value? 1 : -1];
int check1[has_copy_constructor<HasNonConstCopy>::value? 1 : -1];
int check2[has_copy_constructor<HasDeletedCopy>::value? -1 : 1];
int check3[has_copy_constructor<HasPrivateCopy>::value? -1 : 1];
