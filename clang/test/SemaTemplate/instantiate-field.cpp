// RUN: %clang_cc1 -fsyntax-only -verify %s

template<typename T>
struct X {
  int x;
  T y; // expected-error{{data member instantiated with function type}}
  T* z;
  T bitfield : 12; // expected-error{{bit-field 'bitfield' has non-integral type 'float'}} \
                  // expected-error{{data member instantiated with function type}}

  mutable T x2; // expected-error{{data member instantiated with function type}}
};

void test1(const X<int> *xi) {
  int i1 = xi->x;
  const int &i2 = xi->y;
  int* ip1 = xi->z;
  int i3 = xi->bitfield;
  xi->x2 = 17;
}

void test2(const X<float> *xf) {
  (void)xf->x; // expected-note{{in instantiation of template class 'X<float>' requested here}}
}

void test3(const X<int(int)> *xf) {
  (void)xf->x; // expected-note{{in instantiation of template class 'X<int (int)>' requested here}}
}

namespace PR7123 {
  template <class > struct requirement_;

  template <void(*)()> struct instantiate
  { };

  template <class > struct requirement ;
  struct failed ;

  template <class Model> struct requirement<failed *Model::*>
  {
    static void failed()
    {
      ((Model*)0)->~Model(); // expected-note{{in instantiation of}}
    }
  };

  template <class Model> struct requirement_<void(*)(Model)> : requirement<failed *Model::*>
  { };

  template <int> struct Requires_
  { typedef void type; };

  template <class Model> struct usage_requirements
  {
    ~usage_requirements()
    {((Model*)0)->~Model(); } // expected-note{{in instantiation of}}
  };

  template < typename TT > struct BidirectionalIterator
  {
    enum
      { value = 0 };
  
    instantiate< requirement_<void(*)(usage_requirements<BidirectionalIterator>)>::failed> int534; // expected-note{{in instantiation of}}
  
    ~BidirectionalIterator()
    { i--; } // expected-error{{cannot decrement value of type 'PR7123::X'}}
  
    TT i;
  };

  struct X
  { };

  template<typename RanIter> 
  typename Requires_< BidirectionalIterator<RanIter>::value >::type sort(RanIter,RanIter){}

  void f()
  {
    X x;
    sort(x,x);
  }
}

namespace PR7355 {
  template<typename T1> class A {
    class D; // expected-note{{declared here}}
    D d; //expected-error{{implicit instantiation of undefined member 'PR7355::A<int>::D'}}
  };

  A<int> ai; // expected-note{{in instantiation of}}
}

namespace PR8712 {
  template <int dim>
  class B {
  public:
    B(const unsigned char i);
    unsigned char value : (dim > 0 ? dim : 1);
  };

  template <int dim>
  inline B<dim>::B(const unsigned char i) : value(i) {}
}
