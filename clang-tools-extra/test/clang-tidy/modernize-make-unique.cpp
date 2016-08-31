// RUN: %check_clang_tidy %s modernize-make-unique %t

namespace std {

template <typename T>
class default_delete {};

template <typename type, typename Deleter = std::default_delete<type>>
class unique_ptr {
public:
  unique_ptr(type *ptr);
  unique_ptr(const unique_ptr<type> &t) = delete;
  unique_ptr(unique_ptr<type> &&t);
  ~unique_ptr();
  type &operator*() { return *ptr; }
  type *operator->() { return ptr; }
  type *release();
  void reset();
  void reset(type *pt);

private:
  type *ptr;
};

}

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

template<class T> using unique_ptr_ = std::unique_ptr<T>;

void *operator new(__SIZE_TYPE__ Count, void *Ptr);

int g(std::unique_ptr<int> P);

std::unique_ptr<Base> getPointer() {
  return std::unique_ptr<Base>(new Base);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use std::make_unique instead
  // CHECK-FIXES: return std::make_unique<Base>();
}

void basic() {
  std::unique_ptr<int> P1 = std::unique_ptr<int>(new int());
  // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: use std::make_unique instead [modernize-make-unique]
  // CHECK-FIXES: std::unique_ptr<int> P1 = std::make_unique<int>();

  // Without parenthesis.
  std::unique_ptr<int> P2 = std::unique_ptr<int>(new int);
  // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: use std::make_unique instead [modernize-make-unique]
  // CHECK-FIXES: std::unique_ptr<int> P2 = std::make_unique<int>();

  // With auto.
  auto P3 = std::unique_ptr<int>(new int());
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: use std::make_unique instead
  // CHECK-FIXES: auto P3 = std::make_unique<int>();

  {
    // No std.
    using namespace std;
    unique_ptr<int> Q = unique_ptr<int>(new int());
    // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: use std::make_unique instead
    // CHECK-FIXES: unique_ptr<int> Q = std::make_unique<int>();
  }

  std::unique_ptr<int> R(new int());

  // Create the unique_ptr as a parameter to a function.
  int T = g(std::unique_ptr<int>(new int()));
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: use std::make_unique instead
  // CHECK-FIXES: int T = g(std::make_unique<int>());

  // Only replace if the type in the template is the same than the type returned
  // by the new operator.
  auto Pderived = std::unique_ptr<Base>(new Derived());

  // The pointer is returned by the function, nothing to do.
  std::unique_ptr<Base> RetPtr = getPointer();

  // This emulates std::move.
  std::unique_ptr<int> Move = static_cast<std::unique_ptr<int>&&>(P1);

  // Placemenet arguments should not be removed.
  int *PInt = new int;
  std::unique_ptr<int> Placement = std::unique_ptr<int>(new (PInt) int{3});
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
  }
};

void initialization(int T, Base b) {
  // Test different kinds of initialization of the pointee.

  // Direct initialization with parenthesis.
  std::unique_ptr<DPair> PDir1 = std::unique_ptr<DPair>(new DPair(1, T));
  // CHECK-MESSAGES: :[[@LINE-1]]:34: warning: use std::make_unique instead
  // CHECK-FIXES: std::unique_ptr<DPair> PDir1 = std::make_unique<DPair>(1, T);

  // Direct initialization with braces.
  std::unique_ptr<DPair> PDir2 = std::unique_ptr<DPair>(new DPair{2, T});
  // CHECK-MESSAGES: :[[@LINE-1]]:34: warning: use std::make_unique instead
  // CHECK-FIXES: std::unique_ptr<DPair> PDir2 = std::make_unique<DPair>(2, T);

  // Aggregate initialization.
  std::unique_ptr<APair> PAggr = std::unique_ptr<APair>(new APair{T, 1});
  // CHECK-MESSAGES: :[[@LINE-1]]:34: warning: use std::make_unique instead
  // CHECK-FIXES: std::unique_ptr<APair> PAggr = std::make_unique<APair>(APair{T, 1});


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
}

void aliases() {
  typedef std::unique_ptr<int> IntPtr;
  IntPtr Typedef = IntPtr(new int);
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: use std::make_unique instead
  // CHECK-FIXES: IntPtr Typedef = std::make_unique<int>();

  // We use 'bool' instead of '_Bool'.
  typedef std::unique_ptr<bool> BoolPtr;
  BoolPtr BoolType = BoolPtr(new bool);
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: use std::make_unique instead
  // CHECK-FIXES: BoolPtr BoolType = std::make_unique<bool>();

  // We use 'Base' instead of 'struct Base'.
  typedef std::unique_ptr<Base> BasePtr;
  BasePtr StructType = BasePtr(new Base);
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: use std::make_unique instead
  // CHECK-FIXES: BasePtr StructType = std::make_unique<Base>();

#define PTR unique_ptr<int>
  std::unique_ptr<int> Macro = std::PTR(new int);
  // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: use std::make_unique instead
  // CHECK-FIXES: std::unique_ptr<int> Macro = std::make_unique<int>();
#undef PTR

  std::unique_ptr<int> Using = unique_ptr_<int>(new int);
  // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: use std::make_unique instead
  // CHECK-FIXES: std::unique_ptr<int> Using = std::make_unique<int>();
}

void whitespaces() {
  auto Space = std::unique_ptr <int>(new int());
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: use std::make_unique instead
  // CHECK-FIXES: auto Space = std::make_unique<int>();

  auto Spaces = std  ::    unique_ptr  <int>(new int());
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: use std::make_unique instead
  // CHECK-FIXES: auto Spaces = std::make_unique<int>();
}

void nesting() {
  auto Nest = std::unique_ptr<std::unique_ptr<int>>(new std::unique_ptr<int>(new int));
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: use std::make_unique instead
  // CHECK-FIXES: auto Nest = std::make_unique<std::unique_ptr<int>>(new int);
}
