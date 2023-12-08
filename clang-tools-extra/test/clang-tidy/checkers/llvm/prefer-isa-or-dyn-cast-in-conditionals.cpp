// RUN: %check_clang_tidy %s llvm-prefer-isa-or-dyn-cast-in-conditionals %t

struct X;
struct Y;
struct Z {
  int foo();
  X *bar();
  X *cast(Y*);
  bool baz(Y*);
};

template <class X, class Y>
bool isa(Y *);
template <class X, class Y>
X *cast(Y *);
template <class X, class Y>
X *dyn_cast(Y *);
template <class X, class Y>
X *dyn_cast_or_null(Y *);

bool foo(Y *y, Z *z) {
  if (auto x = cast<X>(y))
    return true;
  // CHECK-MESSAGES: :[[@LINE-2]]:16: warning: cast<> in conditional will assert rather than return a null pointer [llvm-prefer-isa-or-dyn-cast-in-conditionals]
  // CHECK-FIXES: if (auto x = dyn_cast<X>(y))

  while (auto x = cast<X>(y))
    break;
  // CHECK-MESSAGES: :[[@LINE-2]]:19: warning: cast<> in conditional
  // CHECK-FIXES: while (auto x = dyn_cast<X>(y))

  if (cast<X>(y))
    return true;
  // CHECK-MESSAGES: :[[@LINE-2]]:7: warning: cast<> in conditional
  // CHECK-FIXES: if (isa<X>(y))

  while (cast<X>(y))
    break;
  // CHECK-MESSAGES: :[[@LINE-2]]:10: warning: cast<> in conditional
  // CHECK-FIXES: while (isa<X>(y))

  do {
    break;
  } while (cast<X>(y));
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: cast<> in conditional
  // CHECK-FIXES: while (isa<X>(y));

  if (dyn_cast<X>(y))
    return true;
  // CHECK-MESSAGES: :[[@LINE-2]]:7: warning: return value from dyn_cast<> not used [llvm-prefer-isa-or-dyn-cast-in-conditionals]
  // CHECK-FIXES: if (isa<X>(y))

  while (dyn_cast<X>(y))
    break;
  // CHECK-MESSAGES: :[[@LINE-2]]:10: warning: return value from dyn_cast<> not used
  // CHECK-FIXES: while (isa<X>(y))

  do {
    break;
  } while (dyn_cast<X>(y));
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: return value from dyn_cast<> not used
  // CHECK-FIXES: while (isa<X>(y));

  if (y && isa<X>(y))
    return true;
  // CHECK-MESSAGES: :[[@LINE-2]]:7: warning: isa_and_nonnull<> is preferred over an explicit test for null followed by calling isa<> [llvm-prefer-isa-or-dyn-cast-in-conditionals]
  // CHECK-FIXES: if (isa_and_nonnull<X>(y))

  if (z->bar() && isa<Y>(z->bar()))
    return true;
  // CHECK-MESSAGES: :[[@LINE-2]]:7: warning:  isa_and_nonnull<> is preferred
  // CHECK-FIXES: if (isa_and_nonnull<Y>(z->bar()))

  if (z->bar() && cast<Y>(z->bar()))
    return true;
  // CHECK-MESSAGES: :[[@LINE-2]]:7: warning: isa_and_nonnull<> is preferred
  // CHECK-FIXES: if (isa_and_nonnull<Y>(z->bar()))

  if (z->bar() && dyn_cast<Y>(z->bar()))
    return true;
  // CHECK-MESSAGES: :[[@LINE-2]]:7: warning: isa_and_nonnull<> is preferred
  // CHECK-FIXES: if (isa_and_nonnull<Y>(z->bar()))

  if (z->bar() && dyn_cast_or_null<Y>(z->bar()))
    return true;
  // CHECK-MESSAGES: :[[@LINE-2]]:7: warning: isa_and_nonnull<> is preferred
  // CHECK-FIXES: if (isa_and_nonnull<Y>(z->bar()))

  bool b = z->bar() && cast<Y>(z->bar());
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: isa_and_nonnull<> is preferred
  // CHECK-FIXES: bool b = isa_and_nonnull<Y>(z->bar());

  // These don't trigger a warning.
  if (auto x = cast<Z>(y)->foo())
    return true;
  if (auto x = z->cast(y))
    return true;
  while (auto x = cast<Z>(y)->foo())
    break;
  if (cast<Z>(y)->foo())
    return true;
  if (z->cast(y))
    return true;
  while (cast<Z>(y)->foo())
    break;
  if (y && cast<X>(z->bar()))
    return true;
  if (z && cast<Z>(y)->foo())
    return true;
  bool b2 = y && cast<X>(z);
  if(z->cast(y))
    return true;
  if (z->baz(cast<Y>(z)))
    return true;

#define CAST(T, Obj) cast<T>(Obj)
#define AUTO_VAR_CAST(X, Y, Z) auto X = cast<Y>(Z)
#define ISA(T, Obj) isa<T>(Obj)
#define ISA_OR_NULL(T, Obj) Obj &&isa<T>(Obj)

  // Macros don't trigger warning.
  if (auto x = CAST(X, y))
    return true;
  if (AUTO_VAR_CAST(x, X, z))
    return true;
  if (z->bar() && ISA(Y, z->bar()))
    return true;
  if (ISA_OR_NULL(Y, z->bar()))
    return true;

  return false;
}
