// RUN: %clang_cc1 -debug-info-kind=limited -triple %itanium_abi_triple -emit-llvm-only %s
// RUN: %clang_cc1 -debug-info-kind=limited -triple %itanium_abi_triple -emit-llvm-only -std=c++98 %s
// RUN: %clang_cc1 -debug-info-kind=limited -triple %itanium_abi_triple -emit-llvm-only -std=c++11 %s
// Check that we don't crash.
// PR12305, PR12315

# 1 "a.h"  3
template < typename T1 > struct Types1
{
  typedef T1 Head;
};
template < typename > struct Types;
template < template < typename > class Tmpl > struct TemplateSel
{
  template < typename T > struct Bind
  {
    typedef Tmpl < T > type;
  };
};
template < typename > struct NoneT;
template < template < typename > class T1, template < typename > class > struct Templates2
{
  typedef TemplateSel < T1 > Head;
};
template < template < typename > class, template < typename > class =
  NoneT, template < typename > class = NoneT, template < typename > class =
  NoneT > struct Templates;
template < template < typename > class T1,
  template < typename > class T2 > struct Templates <T1, T2 >
{
  typedef Templates2 < T1, T2 > type;
};
template < typename T > struct TypeList
{
  typedef Types1 < T > type;
};
template < template < typename > class, class TestSel,
  typename Types > class TypeParameterizedTest
{
public:static bool Register ()
  {
    typedef typename Types::Head Type;
    typename TestSel::template Bind < Type >::type TestClass;
}};

template < template < typename > class Fixture, typename Tests,
  typename Types > class TypeParameterizedTestCase
{
public:static bool Register (char *, char *, int *)
  {
    typedef typename Tests::Head Head;
    TypeParameterizedTest < Fixture, Head, Types >::Register;
}};

template < typename > class TypedTestP1
{
};

namespace gtest_case_TypedTestP1_
{
  template < typename gtest_TypeParam_ > class A:TypedTestP1 <
    gtest_TypeParam_ >
  {
  };
template < typename gtest_TypeParam_ > class B:TypedTestP1 <
    gtest_TypeParam_ >
  {
  };
  typedef Templates < A >::type gtest_AllTests_;
}

template < typename > class TypedTestP2
{
};

namespace gtest_case_TypedTestP2_
{
  template < typename gtest_TypeParam_ > class A:TypedTestP2 <
    gtest_TypeParam_ >
  {
  };
  typedef Templates < A >::type gtest_AllTests_;
}

bool gtest_Int_TypedTestP1 =
  TypeParameterizedTestCase < TypedTestP1,
  gtest_case_TypedTestP1_::gtest_AllTests_,
  TypeList < int >::type >::Register ("Int", "TypedTestP1", 0);
bool gtest_Int_TypedTestP2 =
  TypeParameterizedTestCase < TypedTestP2,
  gtest_case_TypedTestP2_::gtest_AllTests_,
  TypeList < Types < int > >::type >::Register ("Int", "TypedTestP2", 0);

template < typename _Tp > struct new_allocator
{
  typedef _Tp *pointer;
  template < typename > struct rebind {
    typedef new_allocator other;
  };
};
template < typename _Tp > struct allocator:new_allocator < _Tp > {
};
template < typename _Tp, typename _Alloc > struct _Vector_base {
  typedef typename _Alloc::template rebind < _Tp >::other _Tp_alloc_type;
  struct _Vector_impl {
    typename _Tp_alloc_type::pointer _M_end_of_storage;
  };
  _Vector_base () {
    foo((int *) this->_M_impl._M_end_of_storage);
  }
  void foo(int *);
  _Vector_impl _M_impl;
};
template < typename _Tp, typename _Alloc =
allocator < _Tp > >struct vector:_Vector_base < _Tp, _Alloc > { };


template < class T> struct HHH {};
struct DDD { int x_;};
struct Data;
struct X1;
struct CCC:DDD {   virtual void xxx (HHH < X1 >); };
template < class SSS > struct EEE:vector < HHH < SSS > > { };
template < class SSS, class = EEE < SSS > >class FFF { };
template < class SSS, class GGG = EEE < SSS > >class AAA:FFF <GGG> { };
class BBB:virtual CCC {
  void xxx (HHH < X1 >);
  vector < HHH < X1 > >aaa;
};
class ZZZ:AAA < Data >, BBB { virtual ZZZ *ppp () ; };
ZZZ * ZZZ::ppp () { return new ZZZ; }

namespace std
{
  template < class, class > struct pair;
}
namespace __gnu_cxx {
template < typename > class new_allocator;
}
namespace std {
template < typename _Tp > class allocator:__gnu_cxx::new_allocator < _Tp > {
};
template < typename, typename > struct _Vector_base {
};
template < typename _Tp, typename _Alloc = std::allocator < _Tp > >class vector:_Vector_base < _Tp,
  _Alloc
        > {
        };
}

namespace
std {
  template <
      typename,
      typename > struct unary_function;
  template <
      typename,
      typename,
      typename > struct binary_function;
  template <
      typename
      _Tp > struct equal_to:
        binary_function <
        _Tp,
        _Tp,
        bool > {
        };
  template <
      typename
      _Pair > struct _Select1st:
        unary_function <
        _Pair,
        typename
        _Pair::first_type > {
        };
}
# 1 "f.h"  3
using
std::pair;
namespace
__gnu_cxx {
  template <
      class > struct hash;
  template <
      class,
      class,
      class,
      class,
      class
          _EqualKey,
      class >
          class
          hashtable {
           public:
            typedef _EqualKey
                key_equal;
            typedef void key_type;
          };
  using
      std::equal_to;
  using
      std::allocator;
  using
      std::_Select1st;
  template < class _Key, class _Tp, class _HashFn =
      hash < _Key >, class _EqualKey = equal_to < _Key >, class _Alloc =
      allocator < _Tp > >class hash_map {
        typedef
            hashtable <
            pair <
            _Key,
        _Tp >,
        _Key,
        _HashFn,
        _Select1st <
            pair <
            _Key,
        _Tp > >,
        _EqualKey,
        _Alloc >
            _Ht;
       public:
        typedef typename _Ht::key_type key_type;
        typedef typename
            _Ht::key_equal
            key_equal;
      };
}
using
__gnu_cxx::hash_map;
class
C2;
template < class > class scoped_ptr {
};
namespace {
class
    AAA {
protected:
      virtual ~
          AAA () {
          }};
}
template < typename > class EEE;
template < typename CCC, typename =
typename CCC::key_equal, typename =
EEE < CCC > >class III {
};
namespace
util {
  class
      EEE {
      };
}
namespace {
class
    C1:
      util::EEE {
       public:
        class
            C3:
              AAA {
                struct FFF;
                typedef
                    III <
                    hash_map <
                    C2,
                    FFF > >
                        GGG;
                GGG
                    aaa;
                friend
                    C1;
              };
        void
            HHH (C3::GGG &);
      };
}
namespace
n1 {
  class
      Test {
      };
  template <
      typename >
      class
      C7 {
      };
  class
      C4:
        n1::Test {
          vector <
              C1::C3 * >
              a1;
        };
  enum C5 { };
  class
      C6:
        C4,
        n1::C7 <
        C5 > {
        };
  class
      C8:
        C6 {
        };
  class
      C9:
        C8 {
          void
              TestBody ();
        };
  void
      C9::TestBody () {
        scoped_ptr < C1::C3 > context;
      }
}
