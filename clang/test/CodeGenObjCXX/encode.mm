// RUN: %clang_cc1 -std=gnu++98 %s -triple=x86_64-apple-darwin10 -emit-llvm -o - | FileCheck %s

// CHECK: v17@0:8{vector<float, float, float>=}16
// CHECK: {vector<float, float, float>=}
// CHECK: v24@0:816

template <typename T1, typename T2, typename T3> struct vector {
  vector();
  vector(T1,T2,T3);
};

typedef vector< float, float, float > vector3f;

@interface SceneNode
{
 vector3f position;
}

@property (assign, nonatomic) vector3f position;

@end

@interface MyOpenGLView
{
@public
  vector3f position;
}
@property vector3f position;
@end

@implementation MyOpenGLView

@synthesize position;

-(void)awakeFromNib {
 SceneNode *sn;
 vector3f VF3(1.0, 1.0, 1.0);
 [sn setPosition:VF3];
}
@end


class Int3 { int x, y, z; };

// Enforce @encoding for member pointers.
@interface MemPtr {}
- (void) foo: (int (Int3::*)) member;
@end
@implementation MemPtr
- (void) foo: (int (Int3::*)) member {
}
@end

// rdar: // 8519948
typedef float HGVec4f __attribute__ ((vector_size(16)));

@interface RedBalloonHGXFormWrapper {
  HGVec4f m_Transform[4];
}
@end

@implementation RedBalloonHGXFormWrapper
@end

// rdar://9357400
namespace rdar9357400 {
  template<int Dim1 = -1, int Dim2 = -1> struct fixed {
      template<int D> struct rebind { typedef fixed<D> other; };
  };
  
  template<typename Element, int Size>
  class fixed_1D
  {
  public:
      typedef Element value_type;
      typedef value_type array_impl[Size];
    protected:
      array_impl                  m_data;
  };
  
  template<typename Element, typename Alloc>
  class vector;
  
  template<typename Element, int Size>
  class vector< Element, fixed<Size> >
  : public fixed_1D<Element,Size> { };

  typedef vector< float,  fixed<4> > vector4f;

  // CHECK: @_ZN11rdar93574002ggE = constant [49 x i8] c"{vector<float, rdar9357400::fixed<4, -1> >=[4f]}\00"
  extern const char gg[] = @encode(vector4f);
}

// rdar://9624314
namespace rdar9624314 {
  struct B2 { int x; };
  struct B3 {};
  struct S : B2, B3 {};

  // CHECK: @_ZN11rdar96243142ggE = constant [6 x i8] c"{S=i}\00"
  extern const char gg[] = @encode(S);

  struct S2 { unsigned : 0; int x; unsigned : 0; };
  // CHECK: @_ZN11rdar96243142g2E = constant [11 x i8] c"{S2=b0ib0}\00"
  extern const char g2[] = @encode(S2);
}

namespace test {
  class Foo {
  public:
   virtual void f() {};
  };
  
  class Bar {
  public:
   virtual void g() {};
  };
  
  class Zoo : virtual public Foo, virtual public Bar {
  public:
   int x;
   int y;
  };

  // CHECK: @_ZN4test3ecdE = constant [15 x i8] c"{Zoo=^^?ii^^?}\00"
  extern const char ecd[] = @encode(Zoo);
}

struct Base1 {
  char x;
};

struct DBase : public Base1 {
  double x;
  virtual ~DBase();
};

struct Sub_with_virt : virtual DBase {
  long x;
};

struct Sub2 : public Sub_with_virt, public Base1, virtual DBase {
  float x;
};

// CHECK: @g1 = constant [10 x i8] c"{Base1=c}\00"
extern const char g1[] = @encode(Base1);

// CHECK: @g2 = constant [14 x i8] c"{DBase=^^?cd}\00"
extern const char g2[] = @encode(DBase);

// CHECK: @g3 = constant [26 x i8] c"{Sub_with_virt=^^?q^^?cd}\00"
extern const char g3[] = @encode(Sub_with_virt);

// CHECK: @g4 = constant [19 x i8] c"{Sub2=^^?qcf^^?cd}\00"
extern const char g4[] = @encode(Sub2);

// http://llvm.org/PR9927
class allocator {
};
class basic_string     {
struct _Alloc_hider : allocator       {
char* _M_p;
};
_Alloc_hider _M_dataplus;
};

// CHECK: @g5 = constant [32 x i8] c"{basic_string={_Alloc_hider=*}}\00"
extern const char g5[] = @encode(basic_string);


// PR10990
class CefBase {
  virtual ~CefBase() {}
};
class CefBrowser : public virtual CefBase {};
class CefBrowserImpl : public CefBrowser {};
// CHECK: @g6 = constant [21 x i8] c"{CefBrowserImpl=^^?}\00"
extern const char g6[] = @encode(CefBrowserImpl);

// PR10990_2
class CefBase2 {
  virtual ~CefBase2() {}
  int i;
};
class CefBrowser2 : public virtual CefBase2 {};
class CefBrowserImpl2 : public CefBrowser2 {};
// CHECK: @g7 = constant [26 x i8] c"{CefBrowserImpl2=^^?^^?i}\00"
extern const char g7[] = @encode(CefBrowserImpl2);

// <rdar://problem/11324167>
struct Empty {};

struct X : Empty { 
  int array[10];
};

struct Y : Empty {
  X vec;
};

// CHECK: @g8 = constant [14 x i8] c"{Y={X=[10i]}}\00"
extern const char g8[] = @encode(Y);


class dynamic_class {
public:
  virtual ~dynamic_class();
};
@interface has_dynamic_class_ivar
@end
@implementation has_dynamic_class_ivar {
  dynamic_class dynamic_class_ivar;
}
@end
// CHECK: private unnamed_addr constant [41 x i8] c"{dynamic_class=\22_vptr$dynamic_class\22^^?}\00"

namespace PR17142 {
  struct A { virtual ~A(); };
  struct B : virtual A { int y; };
  struct C { virtual ~C(); int z; };
  struct D : C, B { int a; };
  struct E : D {};
  // CHECK: @_ZN7PR171421xE = constant [14 x i8] c"{E=^^?i^^?ii}\00"
  extern const char x[] = @encode(E);
}

// This test used to cause infinite recursion.
template<typename T>
struct S {
  typedef T Ty;
  Ty *t;
};

@interface N
{
  S<N> a;
}
@end

@implementation N
@end

const char *expand_struct() {
  // CHECK: @{{.*}} = private unnamed_addr constant [16 x i8] c"{N={S<N>=^{N}}}\00"
  return @encode(N);
}
