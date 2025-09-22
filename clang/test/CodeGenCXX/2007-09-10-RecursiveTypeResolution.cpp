// RUN: %clang_cc1 -emit-llvm %s -o -
// PR1634

namespace Manta
{
  class CallbackHandle
  {
  protected:virtual ~ CallbackHandle (void)
    {
    }
  };
template < typename Data1 > class CallbackBase_1Data:public CallbackHandle
  {
  };
}

namespace __gnu_cxx
{
  template < typename _Iterator, typename _Container >
  class __normal_iterator
  {
    _Iterator _M_current;
  };
}

namespace std
{
  template < typename _Tp > struct allocator
  {
    typedef _Tp *pointer;
  };
  template < typename _InputIterator,
    typename _Tp > inline void find (_InputIterator __last,
					       const _Tp & __val)
  {
  };
}

namespace Manta
{
  template < typename _Tp, typename _Alloc> struct _Vector_base
  {
    struct _Vector_impl
    {
      _Tp *_M_start;
    };
  public:
    _Vector_impl _M_impl;
  };
  template < typename _Tp, typename _Alloc = std::allocator < _Tp > >
  class vector:protected _Vector_base < _Tp,_Alloc >
  {
  public:
    typedef __gnu_cxx::__normal_iterator < typename _Alloc::pointer,
      vector < _Tp, _Alloc > > iterator;
    iterator end ()
    {
      return {};
    }
  };
  class MantaInterface
  {
  };
  class RTRT
  {
    virtual CallbackHandle *registerTerminationCallback (CallbackBase_1Data <
							 MantaInterface * >*);
    virtual void unregisterCallback (CallbackHandle *);
    typedef vector < CallbackBase_1Data < int >*>PRCallbackMapType;
    PRCallbackMapType parallelPreRenderCallbacks;
  };
}
using namespace Manta;
CallbackHandle *
RTRT::registerTerminationCallback (CallbackBase_1Data < MantaInterface * >*cb)
{
  return cb;
}

void
RTRT::unregisterCallback (CallbackHandle * callback)
{
  {
    typedef CallbackBase_1Data < int > callback_t;
    callback_t *cb = static_cast < callback_t * >(callback);
    find (parallelPreRenderCallbacks.end (), cb);
  }
}
