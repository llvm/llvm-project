// RUN: %clang_cc1 -fsyntax-only %s -std=c++1z -verify

// PR7511
template<a> // expected-error +{{}}
struct int_;

template<a> // expected-error +{{}}
template<int,typename T1,typename>
struct ac
{
  typedef T1 ae
};

template<class>struct aaa
{
  typedef ac<1,int,int>::ae ae // expected-error +{{}}
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
    struct In<a::int_<aaa::a>,my>; // expected-error +{{}}
        
    template<class Event>
    int process(Event)
    {
      In<a::int_<0> > a; // expected-error +{{}}
    }
  } // expected-error +{{}}
  template<class Event>
  int ant(Event)
  {
    region_processing_helper<int>* helper;
    helper->process(0) // expected-error +{{}}
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

int operator<<( basic_ostream<char> , pair ) ; // expected-note +{{}}

void register_object_imp ( )
{
cout << endl<1>; // expected-error +{{}}
}

// PR12933
namespace PR12933 {
  template<typename S> // expected-error +{{}}
    template<typename T>
    void function(S a, T b) {}

  int main() {
    function(0, 1); // expected-error +{{}}
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

namespace var_template_partial_spec_incomplete {
  template<typename T> int n;
  template<typename T, typename U = void> int n<T *>; // expected-error +{{}} expected-note {{}}
  int k = n<void *>;
}

namespace deduceFunctionSpecializationForInvalidOutOfLineFunction {

template <typename InputT, typename OutputT>
struct SourceSelectionRequirement {
  template<typename T>
  OutputT evaluateSelectionRequirement(InputT &&Value) {
  }
};

template <typename InputT, typename OutputT>
OutputT SourceSelectionRequirement<InputT, OutputT>::
evaluateSelectionRequirement<void>(InputT &&Value) { // expected-error {{cannot specialize a member of an unspecialized template}}
  return Value;
}

}

namespace PR51872_part1 {
  template<int> class T1 { template <struct U1> T1(); };
  // expected-error@-1 {{non-type template parameter has incomplete type 'struct U1'}}
  // expected-note@-2  {{forward declaration of 'PR51872_part1::U1'}}

  T1 t1 = 0;
  // expected-error@-1 {{no viable constructor or deduction guide for deduction of template arguments of 'T1'}}
  // expected-note@-6  {{candidate template ignored: could not match 'T1<>' against 'int'}}
}
