/* -*- C++ -*- */
namespace DebugCXX {
  // Records.
  struct Struct {
    int i;
    static int static_member;
  };

  // Enums.
  enum Enum {
    Enumerator
  };
  enum {
    e1 = '1'
  };
  enum {
    e2 = '2'
  };

  // Templates (instantiations).
  template<typename T> struct traits {};
  template<typename T,
           typename Traits = traits<T>
          > class Template {
    T member;
  };
  extern template class Template<int>;

  extern template struct traits<float>;
  typedef class Template<float> FloatInstatiation;

  inline void fn() {
    Template<long> invisible;
  }

  // Non-template inside a template.
  template <class> struct Outer {
    Outer();
    struct Inner {
      Inner(Outer) {}
    };
  };
  template <class T> Outer<T>::Outer() {
    Inner a(*this);
  };

  // Partial template specialization.
  template <typename...> class A;
  template <typename T> class A<T> {};
  typedef A<void> B;
  void foo(B) {}
}

// Virtual class with a forward declaration.
class FwdVirtual;
class FwdVirtual {
  virtual ~FwdVirtual() {}
};

struct PureForwardDecl;

typedef union { int i; } TypedefUnion;
typedef enum { e0 = 0 } TypedefEnum;
typedef struct { int i; } TypedefStruct;

union { int i; } GlobalUnion;
struct { int i; } GlobalStruct;
enum { e5 = 5 } GlobalEnum;

namespace {
  namespace {
    struct InAnonymousNamespace { int i; };
  }
}
