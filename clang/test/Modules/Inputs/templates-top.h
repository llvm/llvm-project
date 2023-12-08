template<typename T> class Vector;

template<typename T> class List {
public:
  void push_back(T);

  struct node {};
  node *head;
  unsigned size;
};

extern List<double> *instantiateListDoubleDeclaration;
extern List<long> *instantiateListLongDeclaration;

namespace A {
  class Y {
    template <typename T> friend class WhereAmI;
  };
}

template <typename T> class A::WhereAmI {
public:
  static void func() {}
};

template<typename T> struct Outer {
  struct Inner {};
};

template<bool, bool> struct ExplicitInstantiation {
  void f() {}
};

template<typename> struct DelayUpdates {};

template<typename T> struct OutOfLineInline {
  void f();
  void g();
  void h();
};
template<typename T> inline void OutOfLineInline<T>::f() {}
template<typename T> inline void OutOfLineInline<T>::g() {}
template<typename T> inline void OutOfLineInline<T>::h() {}

namespace EmitDefaultedSpecialMembers {
  template<typename T> struct SmallVectorImpl {
    SmallVectorImpl() {}
    ~SmallVectorImpl() {} // non-trivial dtor
  };
  template<typename T, unsigned N> struct SmallVector : SmallVectorImpl<T> {
    // trivial dtor
  };
  template<unsigned N> struct SmallString : SmallVector<char, N> {
    // trivial dtor
  };
}

template<typename T> struct WithUndefinedStaticDataMember {
  static T undefined;
};

template<typename T> struct __attribute__((packed, aligned(2))) WithAttributes {
  T value;
};
WithAttributes<int> *get_with_attributes();
