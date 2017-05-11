// RUN: not %clang_cc1 -fsyntax-only %s -std=c++11 2>&1| FileCheck %s

// Note that the error count below doesn't matter. We just want to
// make sure that the parser doesn't crash.
// CHECK: 17 errors

// PR7511
template<a>
struct int_;

template<a>
template<int,typename T1,typename>
struct ac
{
  typedef T1 ae
};

template<class>struct aaa
{
  typedef ac<1,int,int>::ae ae
};

template<class>
struct state_machine
{
  typedef aaa<int>::ae aaa;
  int start()
  {
    ant(0);
  }
  
  template<class>
  struct region_processing_helper
  {
    template<class,int=0>
    struct In;
    
    template<int my>
    struct In<a::int_<aaa::a>,my>;
        
    template<class Event>
    int process(Event)
    {
      In<a::int_<0> > a;
    }
  }
  template<class Event>
  int ant(Event)
  {
    region_processing_helper<int>* helper;
    helper->process(0)
  }
};

int a()
{
  state_machine<int> p;
  p.ant(0);
}

// PR9974
template <int> struct enable_if;
template <class > struct remove_reference ;
template <class _Tp> struct remove_reference<_Tp&> ;

template <class > struct __tuple_like;

template <class _Tp, class _Up, int = __tuple_like<typename remove_reference<_Tp>::type>::value> 
struct __tuple_convertible;

struct pair
{
template<class _Tuple, int = enable_if<__tuple_convertible<_Tuple, pair>::value>::type> 
pair(_Tuple&& );
};

template <class> struct basic_ostream;

template <int> 
void endl( ) ;

extern basic_ostream<char> cout;

int operator<<( basic_ostream<char> , pair ) ;

void register_object_imp ( )
{
cout << endl<1>;
}

// PR12933
namespacae PR12933 {
  template<typename S>
    template<typename T>
    void function(S a, T b) {}

  int main() {
    function(0, 1);
    return 0;
  }
}

// A buildbot failure from libcxx
namespace libcxx_test {
  template <class _Ptr, bool> struct __pointer_traits_element_type;
  template <class _Ptr> struct __pointer_traits_element_type<_Ptr, true>;
  template <template <class, class...> class _Sp, class _Tp, class ..._Args> struct __pointer_traits_element_type<_Sp<_Tp, _Args...>, true> {
    typedef char type;
  };
  template <class T> struct B {};
  __pointer_traits_element_type<B<int>, true>::type x;
}

namespace PR14281_part1 {
  template <class P, int> struct A;
  template <class P> struct A<P, 1>;
  template <template <class, int> class S, class T> struct A<S<T, 1>, 1> {
    typedef char type;
  };
  template <class T, int i> struct B {};
  A<B<int, 1>, 1>::type x;
}

namespace PR14281_part2 {
  typedef decltype(nullptr) nullptr_t;
  template <class P, nullptr_t> struct A;
  template <class P> struct A<P, nullptr>;
  template <template <class, nullptr_t> class S, class T> struct A<S<T, nullptr>, nullptr> {
    typedef char type;
  };
  template <class T, nullptr_t i> struct B {};
  A<B<int, nullptr>, nullptr>::type x;
}

namespace PR14281_part3 {
  extern int some_decl;
  template <class P, int*> struct A;
  template <class P> struct A<P, &some_decl>;
  template <template <class, int*> class S, class T> struct A<S<T, &some_decl>, &some_decl> {
    typedef char type;
  };
  template <class T, int* i> struct B {};
  A<B<int, &some_decl>, &some_decl>::type x;
}
