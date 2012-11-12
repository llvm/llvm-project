// RUN: %clang_cc1 -emit-llvm-only -O1 %s
// This used to crash under asan and valgrind.
// PR12284

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
