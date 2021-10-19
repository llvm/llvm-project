// RUN: %check_clang_tidy %s modernize-make-shared %t -- -- -I %S/Inputs/modernize-smart-ptr

#include "shared_ptr.h"
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

struct Empty {};

template <class T>
using shared_ptr_ = std::shared_ptr<T>;

void *operator new(__SIZE_TYPE__ Count, void *Ptr);

int g(std::shared_ptr<int> P);

std::shared_ptr<Base> getPointer() {
  return std::shared_ptr<Base>(new Base);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use std::make_shared instead
  // CHECK-FIXES: return std::make_shared<Base>();
}

std::shared_ptr<Base> getPointerValue() {
  return std::shared_ptr<Base>(new Base());
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use std::make_shared instead
  // CHECK-FIXES: return std::make_shared<Base>();
}

void basic() {
  std::shared_ptr<int> P1 = std::shared_ptr<int>(new int());
  // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: use std::make_shared instead [modernize-make-shared]
  // CHECK-FIXES: std::shared_ptr<int> P1 = std::make_shared<int>();
  std::shared_ptr<int> P2 = std::shared_ptr<int>(new int);

  P1.reset(new int());
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: use std::make_shared instead [modernize-make-shared]
  // CHECK-FIXES: P1 = std::make_shared<int>();
  P1.reset(new int);

  P1 = std::shared_ptr<int>(new int());
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: use std::make_shared instead [modernize-make-shared]
  // CHECK-FIXES: P1 = std::make_shared<int>();
  P1 = std::shared_ptr<int>(new int);

  // Without parenthesis, default initialization.
  std::shared_ptr<int> P3 = std::shared_ptr<int>(new int);

  P2.reset(new int());
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: use std::make_shared instead [modernize-make-shared]
  // CHECK-FIXES: P2 = std::make_shared<int>();
  P2.reset(new int);

  P2 = std::shared_ptr<int>(new int());
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: use std::make_shared instead [modernize-make-shared]
  // CHECK-FIXES: P2 = std::make_shared<int>();
  P2 = std::shared_ptr<int>(new int);

  // With auto.
  auto P4 = std::shared_ptr<int>(new int());
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: use std::make_shared instead
  // CHECK-FIXES: auto P4 = std::make_shared<int>();
  auto P5 = std::shared_ptr<int>(new int);

  std::shared_ptr<int> P6 = std::shared_ptr<int>((new int()));
  // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: use std::make_shared instead [modernize-make-shared]
  // CHECK-FIXES: std::shared_ptr<int> P6 = std::make_shared<int>();
  std::shared_ptr<int> P7 = std::shared_ptr<int>((new int));

  P4.reset((((new int()))));
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: use std::make_shared instead [modernize-make-shared]
  // CHECK-FIXES: P4 = std::make_shared<int>();
  P4.reset((((new int))));

  P4 = std::shared_ptr<int>(((new int())));
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: use std::make_shared instead [modernize-make-shared]
  // CHECK-FIXES: P4 = std::make_shared<int>();
  P4 = std::shared_ptr<int>(((new int)));

  {
    // No std.
    using namespace std;
    shared_ptr<int> Q = shared_ptr<int>(new int());
    // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: use std::make_shared instead
    // CHECK-FIXES: shared_ptr<int> Q = std::make_shared<int>();
    shared_ptr<int> P = shared_ptr<int>(new int);

    Q = shared_ptr<int>(new int());
    // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use std::make_shared instead
    // CHECK-FIXES: Q = std::make_shared<int>();
    Q = shared_ptr<int>(new int);
  }

  std::shared_ptr<int> R(new int());
  std::shared_ptr<int> S(new int);

  // Create the shared_ptr as a parameter to a function.
  int T = g(std::shared_ptr<int>(new int()));
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: use std::make_shared instead
  // CHECK-FIXES: int T = g(std::make_shared<int>());
  T = g(std::shared_ptr<int>(new int));

  // Only replace if the type in the template is the same as the type returned
  // by the new operator.
  auto Pderived = std::shared_ptr<Base>(new Derived());
  auto PderivedNoparen = std::shared_ptr<Base>(new Derived);

  // OK to replace for reset and assign
  Pderived.reset(new Derived());
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: use std::make_shared instead
  // CHECK-FIXES: Pderived = std::make_shared<Derived>();
  Pderived.reset(new Derived);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: use std::make_shared instead
  // CHECK-FIXES: Pderived = std::make_shared<Derived>();

  Pderived = std::shared_ptr<Derived>(new Derived());
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: use std::make_shared instead
  // CHECK-FIXES: Pderived = std::make_shared<Derived>();
  Pderived = std::shared_ptr<Derived>(new Derived);
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: use std::make_shared instead
  // CHECK-FIXES: Pderived = std::make_shared<Derived>();

  // FIXME: OK to replace if assigned to shared_ptr<Base>
  Pderived = std::shared_ptr<Base>(new Derived());
  Pderived = std::shared_ptr<Base>(new Derived);

  // FIXME: OK to replace when auto is not used
  std::shared_ptr<Base> PBase = std::shared_ptr<Base>(new Derived());
  std::shared_ptr<Base> PBase2 = std::shared_ptr<Base>(new Derived);

  // The pointer is returned by the function, nothing to do.
  std::shared_ptr<Base> RetPtr = getPointer();
  std::shared_ptr<Base> RetPtr2 = getPointerValue();

  // This emulates std::move.
  std::shared_ptr<int> Move = static_cast<std::shared_ptr<int> &&>(P1);

  // Placement arguments should not be removed.
  int *PInt = new int;
  std::shared_ptr<int> Placement = std::shared_ptr<int>(new (PInt) int{3});
  Placement.reset(new (PInt) int{3});
  Placement = std::shared_ptr<int>(new (PInt) int{3});

  std::shared_ptr<int> PlacementNoparen = std::shared_ptr<int>(new (PInt) int);
  PlacementNoparen.reset(new (PInt) int);
  PlacementNoparen = std::shared_ptr<int>(new (PInt) int);
}

// Calling make_smart_ptr from within a member function of a type with a
// private or protected constructor would be ill-formed.
class Private {
private:
  Private(int z) {}

public:
  Private() {}
  void create() {
    auto callsPublic = std::shared_ptr<Private>(new Private);
    // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: use std::make_shared instead
    // CHECK-FIXES: auto callsPublic = std::make_shared<Private>();
    auto ptr = std::shared_ptr<Private>(new Private(42));
    ptr.reset(new Private(42));
    ptr = std::shared_ptr<Private>(new Private(42));
  }

  virtual ~Private();
};

class Protected {
protected:
  Protected() {}

public:
  Protected(int, int) {}
  void create() {
    auto callsPublic = std::shared_ptr<Protected>(new Protected(1, 2));
    // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: use std::make_shared instead
    // CHECK-FIXES: auto callsPublic = std::make_shared<Protected>(1, 2);
    auto ptr = std::shared_ptr<Protected>(new Protected);
    ptr.reset(new Protected);
    ptr = std::shared_ptr<Protected>(new Protected);
  }
};

void initialization(int T, Base b) {
  // Test different kinds of initialization of the pointee.

  // Direct initialization with parenthesis.
  std::shared_ptr<DPair> PDir1 = std::shared_ptr<DPair>(new DPair(1, T));
  // CHECK-MESSAGES: :[[@LINE-1]]:34: warning: use std::make_shared instead
  // CHECK-FIXES: std::shared_ptr<DPair> PDir1 = std::make_shared<DPair>(1, T);
  PDir1.reset(new DPair(1, T));
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use std::make_shared instead
  // CHECK-FIXES: PDir1 = std::make_shared<DPair>(1, T);

  // Direct initialization with braces.
  std::shared_ptr<DPair> PDir2 = std::shared_ptr<DPair>(new DPair{2, T});
  // CHECK-MESSAGES: :[[@LINE-1]]:34: warning: use std::make_shared instead
  // CHECK-FIXES: std::shared_ptr<DPair> PDir2 = std::make_shared<DPair>(2, T);
  PDir2.reset(new DPair{2, T});
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use std::make_shared instead
  // CHECK-FIXES: PDir2 = std::make_shared<DPair>(2, T);

  // Aggregate initialization.
  std::shared_ptr<APair> PAggr = std::shared_ptr<APair>(new APair{T, 1});
  // CHECK-MESSAGES: :[[@LINE-1]]:34: warning: use std::make_shared instead
  // CHECK-FIXES: std::shared_ptr<APair> PAggr = std::make_shared<APair>(APair{T, 1});
  PAggr.reset(new APair{T, 1});
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use std::make_shared instead
  // CHECK-FIXES: std::make_shared<APair>(APair{T, 1});

  // Test different kinds of initialization of the pointee, when the shared_ptr
  // is initialized with braces.

  // Direct initialization with parenthesis.
  std::shared_ptr<DPair> PDir3 = std::shared_ptr<DPair>{new DPair(3, T)};
  // CHECK-MESSAGES: :[[@LINE-1]]:34: warning: use std::make_shared instead
  // CHECK-FIXES: std::shared_ptr<DPair> PDir3 = std::make_shared<DPair>(3, T);

  // Direct initialization with braces.
  std::shared_ptr<DPair> PDir4 = std::shared_ptr<DPair>{new DPair{4, T}};
  // CHECK-MESSAGES: :[[@LINE-1]]:34: warning: use std::make_shared instead
  // CHECK-FIXES: std::shared_ptr<DPair> PDir4 = std::make_shared<DPair>(4, T);

  // Aggregate initialization.
  std::shared_ptr<APair> PAggr2 = std::shared_ptr<APair>{new APair{T, 2}};
  // CHECK-MESSAGES: :[[@LINE-1]]:35: warning: use std::make_shared instead
  // CHECK-FIXES: std::shared_ptr<APair> PAggr2 = std::make_shared<APair>(APair{T, 2});

  // Direct initialization with parenthesis, without arguments.
  std::shared_ptr<DPair> PDir5 = std::shared_ptr<DPair>(new DPair());
  // CHECK-MESSAGES: :[[@LINE-1]]:34: warning: use std::make_shared instead
  // CHECK-FIXES: std::shared_ptr<DPair> PDir5 = std::make_shared<DPair>();

  // Direct initialization with braces, without arguments.
  std::shared_ptr<DPair> PDir6 = std::shared_ptr<DPair>(new DPair{});
  // CHECK-MESSAGES: :[[@LINE-1]]:34: warning: use std::make_shared instead
  // CHECK-FIXES: std::shared_ptr<DPair> PDir6 = std::make_shared<DPair>();

  // Aggregate initialization without arguments.
  std::shared_ptr<Empty> PEmpty = std::shared_ptr<Empty>(new Empty{});
  // CHECK-MESSAGES: :[[@LINE-1]]:35: warning: use std::make_shared instead
  // CHECK-FIXES: std::shared_ptr<Empty> PEmpty = std::make_shared<Empty>(Empty{});
}

void aliases() {
  typedef std::shared_ptr<int> IntPtr;
  IntPtr Typedef = IntPtr(new int());
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: use std::make_shared instead
  // CHECK-FIXES: IntPtr Typedef = std::make_shared<int>();
  IntPtr Typedef2 = IntPtr(new int);

  // We use 'bool' instead of '_Bool'.
  typedef std::shared_ptr<bool> BoolPtr;
  BoolPtr BoolType = BoolPtr(new bool());
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: use std::make_shared instead
  // CHECK-FIXES: BoolPtr BoolType = std::make_shared<bool>();
  BoolPtr BoolType2 = BoolPtr(new bool);

  // We use 'Base' instead of 'struct Base'.
  typedef std::shared_ptr<Base> BasePtr;
  BasePtr StructType = BasePtr(new Base);
// CHECK-MESSAGES: :[[@LINE-1]]:24: warning: use std::make_shared instead
// CHECK-FIXES: BasePtr StructType = std::make_shared<Base>();

#define PTR shared_ptr<int>
  std::shared_ptr<int> Macro = std::PTR(new int());
  // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: use std::make_shared instead
  // CHECK-FIXES: std::shared_ptr<int> Macro = std::make_shared<int>();
  std::shared_ptr<int> Macro2 = std::PTR(new int);
#undef PTR

  std::shared_ptr<int> Using = shared_ptr_<int>(new int());
  // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: use std::make_shared instead
  // CHECK-FIXES: std::shared_ptr<int> Using = std::make_shared<int>();
  std::shared_ptr<int> Using2 = shared_ptr_<int>(new int);
}

void whitespaces() {
  // clang-format off
  auto Space = std::shared_ptr <int>(new int());
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: use std::make_shared instead
  // CHECK-FIXES: auto Space = std::make_shared<int>();
  auto Space2 = std::shared_ptr <int>(new int);

  auto Spaces = std  ::    shared_ptr  <int>(new int());
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: use std::make_shared instead
  // CHECK-FIXES: auto Spaces = std::make_shared<int>();
  auto Spaces2 = std  ::    shared_ptr  <int>(new int);
  // clang-format on
}

void nesting() {
  auto Nest = std::shared_ptr<std::shared_ptr<int>>(new std::shared_ptr<int>(new int));
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: use std::make_shared instead
  // CHECK-FIXES: auto Nest = std::make_shared<std::shared_ptr<int>>(new int);
  Nest.reset(new std::shared_ptr<int>(new int));
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: use std::make_shared instead
  // CHECK-FIXES: Nest = std::make_shared<std::shared_ptr<int>>(new int);
  Nest->reset(new int());
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use std::make_shared instead
  // CHECK-FIXES: *Nest = std::make_shared<int>();
  Nest->reset(new int);
}

void reset() {
  std::shared_ptr<int> P;
  P.reset();
  P.reset(nullptr);
  P.reset(new int());
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use std::make_shared instead
  // CHECK-FIXES: P = std::make_shared<int>();
  P.reset(new int);

  auto Q = &P;
  Q->reset(new int());
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: use std::make_shared instead
  // CHECK-FIXES: *Q = std::make_shared<int>();
  Q->reset(new int);
}
