// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

// <rdar://problem/8124080>
template<typename _Alloc> class allocator;
template<class _CharT> struct char_traits;
template<typename _CharT, typename _Traits = char_traits<_CharT>,            
         typename _Alloc = allocator<_CharT> >
class basic_string;
template<typename _CharT, typename _Traits, typename _Alloc>
const typename basic_string<_CharT, _Traits, _Alloc>::size_type   
basic_string<_CharT, _Traits, _Alloc>::_Rep::_S_max_size // expected-error{{no member named '_Rep' in 'basic_string<_CharT, _Traits, _Alloc>'}}
  = (((npos - sizeof(_Rep_base))/sizeof(_CharT)) - 1) / 4; 

// PR7118
template<typename T>
class Foo {
  class Bar;
  void f() {
    Bar i;
  }
};

// PR7625
template<typename T> struct a : T {
 struct x : T {
   int aa() { return p; } // expected-error{{use of undeclared identifier 'p'}}
 };
};

// rdar://8605381
namespace rdar8605381 {
struct X {};

struct Y { // expected-note{{candidate constructor (the implicit copy constructor) not viable}}
#if __cplusplus >= 201103L // C++11 or later
// expected-note@-2 {{candidate constructor (the implicit move constructor) not viable}}
#endif

  Y();
};

struct {
  Y obj;
} objs[] = {
  new Y // expected-error{{no viable conversion}}
};
}

// http://llvm.org/PR8234
namespace PR8234 {
template<typename Signature>
class callback
{
};

template<typename R , typename ARG_TYPE0>
class callback<R( ARG_TYPE0)>
{
    public:
        callback() {}
};

template< typename ARG_TYPE0>
class callback<void( ARG_TYPE0)>
{
    public:
        callback() {}
};

void f()
{
    callback<void(const int&)> op;
}
}

namespace PR9007 {
  struct bar {
    enum xxx {
      yyy = sizeof(struct foo*)
    };
    foo *xxx();
  };
}

namespace PR9026 {
  class InfallibleTArray {
  };
  class Variant;
  class CompVariant {
    operator const InfallibleTArray&() const;
  };
  class Variant {
    operator const CompVariant&() const;
  };
  void     Write(const Variant& __v);
  void     Write(const InfallibleTArray& __v);
  Variant x;
  void Write2() {
    Write(x);
  }
}

namespace PR10270 {
  template<typename T> class C;
  template<typename T> void f() {
    if (C<T> == 1) // expected-error{{expected unqualified-id}}
      return;
  }
}

namespace rdar11806334 {

class cc_YCbCr;

class cc_rgb
{
 public:
  cc_rgb( uint p ); // expected-error {{unknown type name}}
  cc_rgb( cc_YCbCr v_in );
};

class cc_hsl
{
 public:
  cc_rgb rgb();
  cc_YCbCr YCbCr();
};

class cc_YCbCr
{
 public:
  cc_YCbCr( const cc_rgb v_in );
};

cc_YCbCr cc_hsl::YCbCr()
{
 cc_YCbCr v_out = cc_YCbCr( rgb());
 return v_out;
}

}

namespace test1 {
  int getString(const int*);
  template<int a> class ELFObjectFile  {
    const int* sh;
    ELFObjectFile() {
      switch (*sh) {
      }
      int SectionName(getString(sh));
    }
  };
}

namespace test2 {
  struct fltSemantics ;
  const fltSemantics &foobar();
  void VisitCastExpr(int x) {
    switch (x) {
    case 42:
      const fltSemantics &Sem = foobar();
    }
  }
}

namespace test3 {
  struct nsCSSRect {
  };
  static int nsCSSRect::* sides;
  nsCSSRect dimenX;
  void ParseBoxCornerRadii(int y) {
    switch (y) {
    }
    int& x = dimenX.*sides;
  }
}

namespace pr16964 {
  template<typename> struct bs {
    bs();
    static int* member(); // expected-note{{possible target}}
    member();  // expected-error{{C++ requires a type specifier for all declarations}}
    static member();  // expected-error{{C++ requires a type specifier for all declarations}}
    static int* member(int); // expected-note{{possible target}}
  };

  template<typename T> bs<T>::bs() { member; }  // expected-error{{did you mean to call it}}

  bs<int> test() {
    return bs<int>(); // expected-note{{in instantiation}}
  }
}

namespace pr12791 {
  template<class _Alloc> class allocator {};
  template<class _CharT> struct char_traits;
  struct input_iterator_tag {};
  struct forward_iterator_tag : public input_iterator_tag {};

  template<typename _CharT, typename _Traits, typename _Alloc> struct basic_string {
    struct _Alloc_hider : _Alloc { _Alloc_hider(_CharT*, const _Alloc&); };
    mutable _Alloc_hider _M_dataplus;
    template<class _InputIterator> basic_string(_InputIterator __beg, _InputIterator __end, const _Alloc& __a = _Alloc());
    template<class _InIterator> static _CharT* _S_construct(_InIterator __beg, _InIterator __end, const _Alloc& __a, input_iterator_tag);
    template<class _FwdIterator> static _CharT* _S_construct(_FwdIterator __beg, _FwdIterator __end, const _Alloc& __a, forward_iterator_tag);
    static _CharT* _S_construct(size_type __req, _CharT __c, const _Alloc& __a); // expected-error{{unknown type name 'size_type'}}
  };

  template<typename _CharT, typename _Traits, typename _Alloc>
  template<typename _InputIterator>
  basic_string<_CharT, _Traits, _Alloc>:: basic_string(_InputIterator __beg, _InputIterator __end, const _Alloc& __a)
  : _M_dataplus(_S_construct(__beg, __end, __a, input_iterator_tag()), __a) {}

  template<typename _CharT, typename _Traits = char_traits<_CharT>, typename _Alloc = allocator<_CharT> > struct basic_stringbuf {
    typedef _CharT char_type;
    typedef basic_string<char_type, _Traits, _Alloc> __string_type;
    __string_type str() const {__string_type((char_type*)0,(char_type*)0);}
  };

  template class basic_stringbuf<char>;
}

namespace pr16989 {
  class C {
    template <class T>
    C tpl_mem(T *) { return }    // expected-error{{expected expression}}
    void mem(int *p) {
      tpl_mem(p);
    }
  };
  class C2 {
    void f();
  };
  void C2::f() {}
}

namespace pr20660 {
 appendList(int[]...);     // expected-error {{C++ requires a type specifier for all declarations}}
 appendList(int[]...) { }  // expected-error {{C++ requires a type specifier for all declarations}}
}

