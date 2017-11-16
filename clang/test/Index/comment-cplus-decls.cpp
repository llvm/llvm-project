// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: c-index-test -test-load-source all -comments-xml-schema=%S/../../bindings/xml/comment-xml-schema.rng -target x86_64-apple-darwin10 %s > %t/out
// RUN: FileCheck %s < %t/out
// RUN: c-index-test -test-load-source all -comments-xml-schema=%S/../../bindings/xml/comment-xml-schema.rng -target x86_64-apple-darwin10 -std=c++98 %s > %t/98
// RUN: FileCheck %s < %t/98
// RUN: c-index-test -test-load-source all -comments-xml-schema=%S/../../bindings/xml/comment-xml-schema.rng -target x86_64-apple-darwin10 -std=c++11 %s > %t/11
// RUN: FileCheck %s < %t/11

// Ensure that XML we generate is not invalid.
// RUN: FileCheck %s -check-prefix=WRONG < %t/out
// RUN: FileCheck %s -check-prefix=WRONG < %t/98
// RUN: FileCheck %s -check-prefix=WRONG < %t/11
// WRONG-NOT: CommentXMLInvalid
// rdar://12378714

/**
 * \brief plain c++ class
*/
class Test
{
public:
/**
 * \brief plain c++ constructor
*/
    Test () : reserved (new data()) {}

/**
 * \brief plain c++ member function
*/
    unsigned getID() const
    {
        return reserved->objectID;
    }
/**
 * \brief plain c++ destructor
*/
    ~Test () {}
protected:
    struct data {
        unsigned objectID;
    };
/**
 * \brief plain c++ data field
*/
    data* reserved;
};
// CHECK: <Declaration>class Test {}</Declaration>
// CHECK: <Declaration>Test()</Declaration>
// CHECK: <Declaration>unsigned int getID() const</Declaration>
// CHECK: <Declaration>~Test(){{( noexcept)?}}</Declaration>
// CHECK: <Declaration>Test::data *reserved</Declaration>


class S {
/**
 * \brief Aaa
*/
  friend class Test;
/**
 * \brief Bbb
*/
  friend void foo() {}

/**
 * \brief Ccc
*/
  friend int int_func();

/**
 * \brief Ddd
*/
  friend bool operator==(const Test &, const Test &);

/**
 * \brief Eee
*/
template <typename T> friend void TemplateFriend();

/**
 * \brief Eee
*/
  template <typename T> friend class TemplateFriendClass;

};
// CHECK: <Declaration>friend class Test</Declaration>
// CHECK: <Declaration>friend void foo()</Declaration>
// CHECK: <Declaration>friend int int_func()</Declaration>
// CHECK: <Declaration>friend bool operator==(const Test &amp;, const Test &amp;)</Declaration>
// CHECK: <Declaration>friend template &lt;typename T&gt; void TemplateFriend()</Declaration>
// CHECK: <Declaration>friend template &lt;typename T&gt; class TemplateFriendClass</Declaration>

namespace test0 {
  namespace ns {
    void f(int);
  }

  struct A {
/**
 * \brief Fff
*/
    friend void ns::f(int a);
  };
}
// CHECK: <Declaration>friend void ns::f(int a)</Declaration>

namespace test1 {
  template <class T> struct Outer {
    void foo(T);
    struct Inner {
/**
 * \brief Ggg
*/
      friend void Outer::foo(T);
    };
  };
}
// CHECK: <Declaration>friend void Outer&lt;T&gt;::foo(T)</Declaration>

namespace test2 {
  namespace foo {
    void Func(int x);
  }

  class Bar {
/**
 * \brief Hhh
*/
    friend void ::test2::foo::Func(int x);
  };
}
// CHECK: <Declaration>friend void ::test2::foo::Func(int x)</Declaration>

namespace test3 {
  template<class T> class vector {
   public:
    vector(int i) {}
/**
 * \brief Iii
*/
    void f(const T& t = T()) {}
  };
  class A {
   private:
/**
 * \brief Jjj
*/
    friend void vector<A>::f(const A&);
  };
}
// CHECK: <Declaration>void f(const T &amp;t = T())</Declaration>
// CHECK: <Declaration>friend void vector&lt;A&gt;::f(const test3::A &amp;)</Declaration>

class MyClass
{
/**
 * \brief plain friend test.
*/
  friend class MyClass;
};
// CHECK: <Declaration>friend  class MyClass</Declaration>

template<class _Tp> class valarray
{
private:
/**
 * \brief template friend test.
*/
    template <class T> friend class valarray;
};
// CHECK: <Declaration>template &lt;class T&gt; class valarray</Declaration>
// CHECK: <Declaration>friend template &lt;class T&gt; class valarray</Declaration>

class gslice
{
  valarray<unsigned> __size_;
};
