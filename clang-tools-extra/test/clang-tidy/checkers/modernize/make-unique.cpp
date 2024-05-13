// RUN: %check_clang_tidy -std=c++14-or-later %s modernize-make-unique %t -- -- -I %S/Inputs/smart-ptr

#include "unique_ptr.h"
#include "initializer_list.h"
// CHECK-FIXES: #include <memory>

struct Base {
  Base();
  Base(int, int);
};

struct Derived : public Base {
  Derived();
  Derived(int, int);
};

struct APair {
  int a, b;
};

struct DPair {
  DPair() : a(0), b(0) {}
  DPair(int x, int y) : a(y), b(x) {}
  int a, b;
};

template<typename T>
struct MyVector {
  MyVector(std::initializer_list<T>);
};

struct Empty {};


struct E {
  E(std::initializer_list<int>);
  E();
};

struct F {
  F(std::initializer_list<int>);
  F();
  int a;
};

struct G {
  G(std::initializer_list<int>);
  G(int);
};

struct H {
  H(std::vector<int>);
  H(std::vector<int> &, double);
  H(MyVector<int>, int);
};

struct I {
  I(G);
};

struct J {
  J(E e, int);
};

namespace {
class Foo {};
} // namespace

namespace bar {
class Bar {};
} // namespace bar

template <class T>
using unique_ptr_ = std::unique_ptr<T>;

void *operator new(__SIZE_TYPE__ Count, void *Ptr);

int g(std::unique_ptr<int> P);

std::unique_ptr<Base> getPointer() {
  return std::unique_ptr<Base>(new Base);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use std::make_unique instead
  // CHECK-FIXES: return std::make_unique<Base>();
}

std::unique_ptr<Base> getPointerValue() {
  return std::unique_ptr<Base>(new Base());
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use std::make_unique instead
  // CHECK-FIXES: return std::make_unique<Base>();
}

void basic() {
  std::unique_ptr<int> P1 = std::unique_ptr<int>(new int());
  // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: use std::make_unique instead [modernize-make-unique]
  // CHECK-FIXES: std::unique_ptr<int> P1 = std::make_unique<int>();
  std::unique_ptr<int> P2 = std::unique_ptr<int>(new int);

  P1.reset(new int());
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: use std::make_unique instead [modernize-make-unique]
  // CHECK-FIXES: P1 = std::make_unique<int>();
  P2.reset(new int);

  P1 = std::unique_ptr<int>(new int());
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: use std::make_unique instead [modernize-make-unique]
  // CHECK-FIXES: P1 = std::make_unique<int>();
  P1 = std::unique_ptr<int>(new int);

  // Without parenthesis, default initialization.
  std::unique_ptr<int> P3 = std::unique_ptr<int>(new int);

  P2.reset(new int);

  P2 = std::unique_ptr<int>(new int);

  // With auto.
  auto P4 = std::unique_ptr<int>(new int());
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: use std::make_unique instead
  // CHECK-FIXES: auto P4 = std::make_unique<int>();
  auto P5 = std::unique_ptr<int>(new int);

  std::unique_ptr<int> P6 = std::unique_ptr<int>((new int()));
  // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: use std::make_unique instead [modernize-make-unique]
  // CHECK-FIXES: std::unique_ptr<int> P6 = std::make_unique<int>();
  std::unique_ptr<int> P7 = std::unique_ptr<int>((new int));

  P4.reset((new int()));
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: use std::make_unique instead [modernize-make-unique]
  // CHECK-FIXES: P4 = std::make_unique<int>();
  P5.reset((new int));

  std::unique_ptr<int> P8 = std::unique_ptr<int>((((new int()))));
  // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: use std::make_unique instead [modernize-make-unique]
  // CHECK-FIXES: std::unique_ptr<int> P8 = std::make_unique<int>();
  std::unique_ptr<int> P9 = std::unique_ptr<int>((((new int))));

  P5.reset(((((new int())))));
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: use std::make_unique instead [modernize-make-unique]
  // CHECK-FIXES: P5 = std::make_unique<int>();
  P6.reset(((((new int)))));

  {
    // No std.
    using namespace std;
    unique_ptr<int> Q = unique_ptr<int>(new int());
    // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: use std::make_unique instead
    // CHECK-FIXES: unique_ptr<int> Q = std::make_unique<int>();
    unique_ptr<int> P = unique_ptr<int>(new int);

    Q = unique_ptr<int>(new int());
    // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use std::make_unique instead
    // CHECK-FIXES: Q = std::make_unique<int>();

    P = unique_ptr<int>(new int);
  }

  std::unique_ptr<int> R(new int());
  std::unique_ptr<int> S(new int);

  // Create the unique_ptr as a parameter to a function.
  int T = g(std::unique_ptr<int>(new int()));
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: use std::make_unique instead
  // CHECK-FIXES: int T = g(std::make_unique<int>());
  T = g(std::unique_ptr<int>(new int));

  // Only replace if the type in the template is the same as the type returned
  // by the new operator.
  auto Pderived = std::unique_ptr<Base>(new Derived());
  auto PderivedNoparen = std::unique_ptr<Base>(new Derived);

  // OK to replace for reset and assign
  Pderived.reset(new Derived());
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: use std::make_unique instead
  // CHECK-FIXES: Pderived = std::make_unique<Derived>();
  PderivedNoparen.reset(new Derived);
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: use std::make_unique instead
  // CHECK-FIXES: PderivedNoparen = std::make_unique<Derived>();

  Pderived = std::unique_ptr<Derived>(new Derived());
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: use std::make_unique instead
  // CHECK-FIXES: Pderived = std::make_unique<Derived>();
  PderivedNoparen = std::unique_ptr<Derived>(new Derived);
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: use std::make_unique instead
  // CHECK-FIXES: PderivedNoparen = std::make_unique<Derived>();

  // FIXME: OK to replace if assigned to unique_ptr<Base>
  Pderived = std::unique_ptr<Base>(new Derived());
  Pderived = std::unique_ptr<Base>(new Derived);

  // FIXME: OK to replace when auto is not used
  std::unique_ptr<Base> PBase = std::unique_ptr<Base>(new Derived());
  std::unique_ptr<Base> PBaseNoparen = std::unique_ptr<Base>(new Derived);

  // The pointer is returned by the function, nothing to do.
  std::unique_ptr<Base> RetPtr = getPointer();
  RetPtr = getPointerValue();

  // This emulates std::move.
  std::unique_ptr<int> Move = static_cast<std::unique_ptr<int> &&>(P1);

  // Placement arguments should not be removed.
  int *PInt = new int;
  std::unique_ptr<int> Placement = std::unique_ptr<int>(new (PInt) int{3});
  Placement.reset(new (PInt) int{3});
  Placement = std::unique_ptr<int>(new (PInt) int{3});

  std::unique_ptr<int> PlacementNoparen = std::unique_ptr<int>(new (PInt) int);
  PlacementNoparen.reset(new (PInt) int);
  PlacementNoparen = std::unique_ptr<int>(new (PInt) int);
}

// Calling make_smart_ptr from within a member function of a type with a
// private or protected constructor would be ill-formed.
class Private {
private:
  Private(int z) {}

public:
  Private() {}
  void create() {
    auto callsPublic = std::unique_ptr<Private>(new Private);
    // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: use std::make_unique instead
    // CHECK-FIXES: auto callsPublic = std::make_unique<Private>();
    auto ptr = std::unique_ptr<Private>(new Private(42));
    ptr.reset(new Private(42));
    ptr = std::unique_ptr<Private>(new Private(42));
  }

  virtual ~Private();
};

class Protected {
protected:
  Protected() {}

public:
  Protected(int, int) {}
  void create() {
    auto callsPublic = std::unique_ptr<Protected>(new Protected(1, 2));
    // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: use std::make_unique instead
    // CHECK-FIXES: auto callsPublic = std::make_unique<Protected>(1, 2);
    auto ptr = std::unique_ptr<Protected>(new Protected);
    ptr.reset(new Protected);
    ptr = std::unique_ptr<Protected>(new Protected);
  }
};

void initialization(int T, Base b) {
  // Test different kinds of initialization of the pointee.

  // Direct initialization with parenthesis.
  std::unique_ptr<DPair> PDir1 = std::unique_ptr<DPair>(new DPair(1, T));
  // CHECK-MESSAGES: :[[@LINE-1]]:34: warning: use std::make_unique instead
  // CHECK-FIXES: std::unique_ptr<DPair> PDir1 = std::make_unique<DPair>(1, T);
  PDir1.reset(new DPair(1, T));
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use std::make_unique instead
  // CHECK-FIXES: PDir1 = std::make_unique<DPair>(1, T);

  // Direct initialization with braces.
  std::unique_ptr<DPair> PDir2 = std::unique_ptr<DPair>(new DPair{2, T});
  // CHECK-MESSAGES: :[[@LINE-1]]:34: warning: use std::make_unique instead
  // CHECK-FIXES: std::unique_ptr<DPair> PDir2 = std::make_unique<DPair>(2, T);
  PDir2.reset(new DPair{2, T});
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use std::make_unique instead
  // CHECK-FIXES: PDir2 = std::make_unique<DPair>(2, T);

  // Aggregate initialization.
  std::unique_ptr<APair> PAggr = std::unique_ptr<APair>(new APair{T, 1});
  // CHECK-MESSAGES: :[[@LINE-1]]:34: warning: use std::make_unique instead
  // CHECK-FIXES: std::unique_ptr<APair> PAggr = std::make_unique<APair>(APair{T, 1});
  PAggr.reset(new APair{T, 1});
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use std::make_unique instead
  // CHECK-FIXES: std::make_unique<APair>(APair{T, 1});

  // Check aggregate init with intermediate temporaries.
  std::unique_ptr<APair> PAggrTemp = std::unique_ptr<APair>(new APair({T, 1}));
  // CHECK-MESSAGES: :[[@LINE-1]]:38: warning: use std::make_unique instead
  // CHECK-FIXES: std::unique_ptr<APair> PAggrTemp = std::unique_ptr<APair>(new APair({T, 1}));
  PAggrTemp.reset(new APair({T, 1}));
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: use std::make_unique instead
  // CHECK-FIXES: PAggrTemp.reset(new APair({T, 1}));

  // Test different kinds of initialization of the pointee, when the unique_ptr
  // is initialized with braces.

  // Direct initialization with parenthesis.
  std::unique_ptr<DPair> PDir3 = std::unique_ptr<DPair>{new DPair(3, T)};
  // CHECK-MESSAGES: :[[@LINE-1]]:34: warning: use std::make_unique instead
  // CHECK-FIXES: std::unique_ptr<DPair> PDir3 = std::make_unique<DPair>(3, T);

  // Direct initialization with braces.
  std::unique_ptr<DPair> PDir4 = std::unique_ptr<DPair>{new DPair{4, T}};
  // CHECK-MESSAGES: :[[@LINE-1]]:34: warning: use std::make_unique instead
  // CHECK-FIXES: std::unique_ptr<DPair> PDir4 = std::make_unique<DPair>(4, T);

  // Aggregate initialization.
  std::unique_ptr<APair> PAggr2 = std::unique_ptr<APair>{new APair{T, 2}};
  // CHECK-MESSAGES: :[[@LINE-1]]:35: warning: use std::make_unique instead
  // CHECK-FIXES: std::unique_ptr<APair> PAggr2 = std::make_unique<APair>(APair{T, 2});

  // Direct initialization with parenthesis, without arguments.
  std::unique_ptr<DPair> PDir5 = std::unique_ptr<DPair>(new DPair());
  // CHECK-MESSAGES: :[[@LINE-1]]:34: warning: use std::make_unique instead
  // CHECK-FIXES: std::unique_ptr<DPair> PDir5 = std::make_unique<DPair>();

  // Direct initialization with braces, without arguments.
  std::unique_ptr<DPair> PDir6 = std::unique_ptr<DPair>(new DPair{});
  // CHECK-MESSAGES: :[[@LINE-1]]:34: warning: use std::make_unique instead
  // CHECK-FIXES: std::unique_ptr<DPair> PDir6 = std::make_unique<DPair>();

  // Aggregate initialization without arguments.
  std::unique_ptr<Empty> PEmpty = std::unique_ptr<Empty>(new Empty{});
  // CHECK-MESSAGES: :[[@LINE-1]]:35: warning: use std::make_unique instead
  // CHECK-FIXES: std::unique_ptr<Empty> PEmpty = std::make_unique<Empty>(Empty{});

  // Initialization with default constructor.
  std::unique_ptr<E> PE1 = std::unique_ptr<E>(new E{});
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: use std::make_unique instead
  // CHECK-FIXES: std::unique_ptr<E> PE1 = std::make_unique<E>();
  PE1.reset(new E{});
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use std::make_unique instead
  // CHECK-FIXES: PE1 = std::make_unique<E>();

  // No warnings for `auto` new expression.
  PE1.reset(new auto(E()));

  //============================================================================
  //  NOTE: For initializer-list constructors, the check only gives warnings,
  //  and no fixes are generated.
  //============================================================================

  // Initialization with the initializer-list constructor.
  std::unique_ptr<E> PE2 = std::unique_ptr<E>(new E{1, 2});
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: use std::make_unique instead
  // CHECK-FIXES: std::unique_ptr<E> PE2 = std::unique_ptr<E>(new E{1, 2});
  PE2.reset(new E{1, 2});
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use std::make_unique instead
  // CHECK-FIXES: PE2.reset(new E{1, 2});

  // Initialization with default constructor.
  std::unique_ptr<F> PF1 = std::unique_ptr<F>(new F());
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: use std::make_unique instead
  // CHECK-FIXES: std::unique_ptr<F> PF1 = std::make_unique<F>();
  PF1.reset(new F());
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use std::make_unique instead
  // CHECK-FIXES: PF1 = std::make_unique<F>();

  // Initialization with default constructor.
  std::unique_ptr<F> PF2 = std::unique_ptr<F>(new F{});
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: use std::make_unique instead
  // CHECK-FIXES: std::unique_ptr<F> PF2 = std::make_unique<F>();
  PF2.reset(new F());
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use std::make_unique instead
  // CHECK-FIXES: PF2 = std::make_unique<F>();

  // Initialization with the initializer-list constructor.
  std::unique_ptr<F> PF3 = std::unique_ptr<F>(new F{1});
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: use std::make_unique instead
  // CHECK-FIXES: std::unique_ptr<F> PF3 = std::unique_ptr<F>(new F{1});
  PF3.reset(new F{1});
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use std::make_unique instead
  // CHECK-FIXES: PF3.reset(new F{1});

  // Initialization with the initializer-list constructor.
  std::unique_ptr<F> PF4 = std::unique_ptr<F>(new F{1, 2});
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: use std::make_unique instead
  // CHECK-FIXES: std::unique_ptr<F> PF4 = std::unique_ptr<F>(new F{1, 2});

  // Initialization with the initializer-list constructor.
  std::unique_ptr<F> PF5 = std::unique_ptr<F>(new F({1, 2}));
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: use std::make_unique instead
  // CHECK-FIXES: std::unique_ptr<F> PF5 = std::unique_ptr<F>(new F({1, 2}));

  // Initialization with the initializer-list constructor as the default
  // constructor is not present.
  std::unique_ptr<G> PG1 = std::unique_ptr<G>(new G{});
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: use std::make_unique instead
  // CHECK-FIXES: std::unique_ptr<G> PG1 = std::unique_ptr<G>(new G{});
  PG1.reset(new G{});
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use std::make_unique instead
  // CHECK-FIXES: PG1.reset(new G{});

  // Initialization with the initializer-list constructor.
  std::unique_ptr<G> PG2 = std::unique_ptr<G>(new G{1});
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: use std::make_unique instead
  // CHECK-FIXES: std::unique_ptr<G> PG2 = std::unique_ptr<G>(new G{1});

  // Initialization with the initializer-list constructor.
  std::unique_ptr<G> PG3 = std::unique_ptr<G>(new G{1, 2});
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: use std::make_unique instead
  // CHECK-FIXES: std::unique_ptr<G> PG3 = std::unique_ptr<G>(new G{1, 2});

  std::unique_ptr<H> PH1 = std::unique_ptr<H>(new H({1, 2, 3}));
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: use std::make_unique instead
  // CHECK-FIXES: std::unique_ptr<H> PH1 = std::unique_ptr<H>(new H({1, 2, 3}));
  PH1.reset(new H({1, 2, 3}));
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use std::make_unique instead
  // CHECK-FIXES: PH1.reset(new H({1, 2, 3}));

  std::unique_ptr<H> PH2 = std::unique_ptr<H>(new H({1, 2, 3}, 1));
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: use std::make_unique instead
  // CHECK-FIXES: std::unique_ptr<H> PH2 = std::unique_ptr<H>(new H({1, 2, 3}, 1));
  PH2.reset(new H({1, 2, 3}, 1));
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use std::make_unique instead
  // CHECK-FIXES: PH2.reset(new H({1, 2, 3}, 1));

  std::unique_ptr<H> PH3 = std::unique_ptr<H>(new H({1, 2, 3}, 1.0));
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: use std::make_unique instead
  // CHECK-FIXES: std::unique_ptr<H> PH3 = std::unique_ptr<H>(new H({1, 2, 3}, 1.0));
  PH3.reset(new H({1, 2, 3}, 1.0));
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use std::make_unique instead
  // CHECK-FIXES: PH3.reset(new H({1, 2, 3}, 1.0));

  std::unique_ptr<I> PI1 = std::unique_ptr<I>(new I(G({1, 2, 3})));
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: use std::make_unique instead
  // CHECK-FIXES: std::unique_ptr<I> PI1 = std::make_unique<I>(G({1, 2, 3}));
  PI1.reset(new I(G({1, 2, 3})));
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use std::make_unique instead
  // CHECK-FIXES: PI1 = std::make_unique<I>(G({1, 2, 3}));

  std::unique_ptr<J> PJ1 = std::unique_ptr<J>(new J({1, 2}, 1));
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: use std::make_unique instead
  // CHECK-FIXES: std::unique_ptr<J> PJ1 = std::unique_ptr<J>(new J({1, 2}, 1));
  PJ1.reset(new J({1, 2}, 1));
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use std::make_unique instead
  // CHECK-FIXES: PJ1.reset(new J({1, 2}, 1));

  std::unique_ptr<J> PJ2 = std::unique_ptr<J>(new J(E{1, 2}, 1));
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: use std::make_unique instead
  // CHECK-FIXES: std::unique_ptr<J> PJ2 = std::unique_ptr<J>(new J(E{1, 2}, 1));
  PJ2.reset(new J(E{1, 2}, 1));
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use std::make_unique instead
  // CHECK-FIXES: PJ2.reset(new J(E{1, 2}, 1));

  std::unique_ptr<J> PJ3 = std::unique_ptr<J>(new J{ {1, 2}, 1 });
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: use std::make_unique instead
  // CHECK-FIXES: std::unique_ptr<J> PJ3 = std::unique_ptr<J>(new J{ {1, 2}, 1 });
  PJ3.reset(new J{ {1, 2}, 1 });
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use std::make_unique instead
  // CHECK-FIXES:  PJ3.reset(new J{ {1, 2}, 1 });

  std::unique_ptr<J> PJ4 = std::unique_ptr<J>(new J{E{1, 2}, 1});
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: use std::make_unique instead
  // CHECK-FIXES: std::unique_ptr<J> PJ4 = std::unique_ptr<J>(new J{E{1, 2}, 1});
  PJ4.reset(new J{E{1, 2}, 1});
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use std::make_unique instead
  // CHECK-FIXES: PJ4.reset(new J{E{1, 2}, 1});

  std::unique_ptr<Foo> FF = std::unique_ptr<Foo>(new Foo());
  // CHECK-MESSAGES: :[[@LINE-1]]:29: warning:
  // CHECK-FIXES: std::unique_ptr<Foo> FF = std::make_unique<Foo>();
  FF.reset(new Foo());
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning:
  // CHECK-FIXES: FF = std::make_unique<Foo>();

  std::unique_ptr<bar::Bar> BB = std::unique_ptr<bar::Bar>(new bar::Bar());
  // CHECK-MESSAGES: :[[@LINE-1]]:34: warning:
  // CHECK-FIXES: std::unique_ptr<bar::Bar> BB = std::make_unique<bar::Bar>();
  BB.reset(new bar::Bar());
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning:
  // CHECK-FIXES: BB = std::make_unique<bar::Bar>();

  std::unique_ptr<Foo[]> FFs;
  FFs.reset(new Foo[5]);
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning:
  // CHECK-FIXES: FFs = std::make_unique<Foo[]>(5);
  FFs.reset(new Foo[5]());
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning:
  // CHECK-FIXES: FFs = std::make_unique<Foo[]>(5);
  const int Num = 1;
  FFs.reset(new Foo[Num]);
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning:
  // CHECK-FIXES: FFs = std::make_unique<Foo[]>(Num);
  int Num2 = 1;
  FFs.reset(new Foo[Num2]);
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning:
  // CHECK-FIXES: FFs = std::make_unique<Foo[]>(Num2);

  std::unique_ptr<int[]> FI;
  FI.reset(new int[5]()); // value initialization.
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning:
  // CHECK-FIXES: FI = std::make_unique<int[]>(5);

  // The check doesn't give warnings and fixes for cases where the original new
  // expression does default initialization.
  FI.reset(new int[5]);
  FI.reset(new int[Num]);
  FI.reset(new int[Num2]);
}

void aliases() {
  typedef std::unique_ptr<int> IntPtr;
  IntPtr Typedef = IntPtr(new int());
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: use std::make_unique instead
  // CHECK-FIXES: IntPtr Typedef = std::make_unique<int>();
  IntPtr Typedef2 = IntPtr(new int);

  // We use 'bool' instead of '_Bool'.
  typedef std::unique_ptr<bool> BoolPtr;
  BoolPtr BoolType = BoolPtr(new bool());
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: use std::make_unique instead
  // CHECK-FIXES: BoolPtr BoolType = std::make_unique<bool>();
  BoolPtr BoolType2 = BoolPtr(new bool);

  // We use 'Base' instead of 'struct Base'.
  typedef std::unique_ptr<Base> BasePtr;
  BasePtr StructType = BasePtr(new Base);
// CHECK-MESSAGES: :[[@LINE-1]]:24: warning: use std::make_unique instead
// CHECK-FIXES: BasePtr StructType = std::make_unique<Base>();

#define PTR unique_ptr<int>
  std::unique_ptr<int> Macro = std::PTR(new int());
  // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: use std::make_unique instead
  // CHECK-FIXES: std::unique_ptr<int> Macro = std::make_unique<int>();
  std::unique_ptr<int> Macro2 = std::PTR(new int);
#undef PTR

  std::unique_ptr<int> Using = unique_ptr_<int>(new int());
  // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: use std::make_unique instead
  // CHECK-FIXES: std::unique_ptr<int> Using = std::make_unique<int>();
  std::unique_ptr<int> Using2 = unique_ptr_<int>(new int);
}

void whitespaces() {
  // clang-format off
  auto Space = std::unique_ptr <int>(new int());
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: use std::make_unique instead
  // CHECK-FIXES: auto Space = std::make_unique<int>();
  auto Space2 = std::unique_ptr <int>(new int);

  auto Spaces = std  ::    unique_ptr  <int>(new int());
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: use std::make_unique instead
  // CHECK-FIXES: auto Spaces = std::make_unique<int>();
  auto Spaces2 = std  ::    unique_ptr  <int>(new int);
  // clang-format on
}

void nesting() {
  auto Nest = std::unique_ptr<std::unique_ptr<int>>(new std::unique_ptr<int>(new int));
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: use std::make_unique instead
  // CHECK-FIXES: auto Nest = std::make_unique<std::unique_ptr<int>>(new int);
  Nest.reset(new std::unique_ptr<int>(new int));
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: use std::make_unique instead
  // CHECK-FIXES: Nest = std::make_unique<std::unique_ptr<int>>(new int);
  Nest->reset(new int());
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use std::make_unique instead
  // CHECK-FIXES: *Nest = std::make_unique<int>();
}

void reset() {
  std::unique_ptr<int> P;
  P.reset();
  P.reset(nullptr);
  P.reset(new int());
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use std::make_unique instead
  // CHECK-FIXES: P = std::make_unique<int>();
  P.reset(new int);

  auto Q = &P;
  Q->reset(new int());
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: use std::make_unique instead
  // CHECK-FIXES: *Q = std::make_unique<int>();
  Q->reset(new int);
}

#define DEFINE(...) __VA_ARGS__
template<typename T>
void g2(std::unique_ptr<Foo> *t) {
  DEFINE(auto p = std::unique_ptr<Foo>(new Foo); t->reset(new Foo););
}
void macro() {
  std::unique_ptr<Foo> *t;
  g2<bar::Bar>(t);
}
#undef DEFINE

class UniqueFoo : public std::unique_ptr<Foo> {
 public:
  void foo() {
    reset(new Foo);
    this->reset(new Foo);
    // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: use std::make_unique instead
    // CHECK-FIXES: *this = std::make_unique<Foo>();
    this->reset(new Foo());
    // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: use std::make_unique instead
    // CHECK-FIXES: *this = std::make_unique<Foo>();
    (*this).reset(new Foo);
    // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: use std::make_unique instead
    // CHECK-FIXES: (*this) = std::make_unique<Foo>();
    (*this).reset(new Foo());
    // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: use std::make_unique instead
    // CHECK-FIXES: (*this) = std::make_unique<Foo>();
  }
};

// Ignore statements inside a template instantiation.
template<typename T>
void template_fun(T* t) {
  std::unique_ptr<T> t2 = std::unique_ptr<T>(new T);
  std::unique_ptr<T> t3 = std::unique_ptr<T>(new T());
  t2.reset(new T);
  t3.reset(new T());
}

void invoke_template() {
  Foo* foo;
  template_fun(foo);
}

void no_fix_for_invalid_new_loc() {
  // FIXME: Although the code is valid, the end location of `new struct Base` is
  // invalid. Correct it once https://bugs.llvm.org/show_bug.cgi?id=35952 is
  // fixed.
  auto T = std::unique_ptr<Base>(new struct Base);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: use std::make_unique instead
  // CHECK-FIXES: auto T = std::unique_ptr<Base>(new struct Base);
}
