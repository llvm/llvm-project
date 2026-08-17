# Available Checkers

The analyzer performs checks that are categorized into families or "checkers".

The default set of checkers covers a variety of checks targeted at finding security and API usage bugs,
dead code, and other logic errors. See the {ref}`default-checkers` checkers list below.

In addition to these, the analyzer contains a number of {ref}`alpha-checkers` (aka *alpha* checkers).
These checkers are under development and are switched off by default. They may crash or emit a higher number of false positives.

The {ref}`debug-checkers` package contains checkers for analyzer developers for debugging purposes.

```{contents} Table of Contents
:depth: 4
```

(default-checkers)=

## Default Checkers

(core-checkers)=

### core

Models core language features and contains general-purpose checkers such as division by zero,
null pointer dereference, usage of uninitialized values, etc.
*These checkers must be always switched on as other checker rely on them.*

(core-bitwiseshift)=

#### core.BitwiseShift (C, C++)

Finds undefined behavior caused by the bitwise left- and right-shift operator
operating on integer types.

By default, this checker only reports situations when the right operand is
either negative or larger than the bit width of the type of the left operand;
these are logically unsound.

Moreover, if the pedantic mode is activated by
`-analyzer-config core.BitwiseShift:Pedantic=true`, then this checker also
reports situations where the *left* operand of a shift operator is negative or
overflow occurs during the right shift of a signed value. (Most compilers
handle these predictably, but the C standard and the C++ standards before C++20
say that they're undefined behavior. In the C++20 standard these constructs are
well-defined, so activating pedantic mode in C++20 has no effect.)

**Examples**

```cpp
static_assert(sizeof(int) == 4, "assuming 32-bit int")

void basic_examples(int a, int b) {
  if (b < 0) {
    b = a << b; // warn: right operand is negative in left shift
  } else if (b >= 32) {
    b = a >> b; // warn: right shift overflows the capacity of 'int'
  }
}

int pedantic_examples(int a, int b) {
  if (a < 0) {
    return a >> b; // warn: left operand is negative in right shift
  }
  a = 1000u << 31; // OK, overflow of unsigned value is well-defined, a == 0
  if (b > 10) {
    a = b << 31; // this is undefined before C++20, but the checker doesn't
                 // warn because it doesn't know the exact value of b
  }
  return 1000 << 31; // warn: this overflows the capacity of 'int'
}
```

**Solution**

Ensure the shift operands are in proper range before shifting.

(core-callandmessage)=

#### core.CallAndMessage (C, C++, ObjC)

Check for logical errors for function calls and Objective-C message expressions
(e.g., uninitialized arguments, null function pointers).

This checker is a collection of related checks that are controlled by checker
options. The following checks are all enabled by default, but can be turned off
by setting their option to `false`:

- **FunctionPointer** Check for null or undefined function pointer at function
  call.
- **CXXThisMethodCall** Check for null or undefined `this` pointer at method
  call.
- **CXXDeallocationArg** Check for null or undefined argument of
  `operator delete`.
- **ArgInitializedness** Check for undefined pass-by-value function arguments.
- **ParameterCount** Check for correct number of passed arguments to functions
  or ObjC blocks. This will warn if the actual argument count is less (but not
  if more) than the required count (by the declaration).
- **NilReceiver** Check whether the receiver in a message expression is
  `nil`.
- **UndefReceiver** Check whether the receiver in a message expression is
  undefined.

The following check is disabled by default (because it is more likely to
produce false positives), this can be turned on by set the option to `true`:

- **ArgPointeeInitializedness** Check for undefined pass-by-reference (pointer
  to constant value or constant reference) function arguments. In special cases
  non-constant arguments are checked. This happens for C library functions
  where it is required to initialize (at least partially) a passed structure
  which is used for both input and output (for example last argument of
  `mktime` or `mbrlen`).

**Additional options**

- **ArgPointeeInitializednessComplete** Controls when to emit the warning at
  the **ArgPointeeInitializedness** check. If this option is `false` (the
  default), a `struct` is considered to be "initialized" when at least one
  member is initialized. When this option is set to `true`, structures are
  only accepted as initialized when all members are initialized. (Arguments of
  C library functions which require initialization are always checked as if the
  option would be `false`.)

**Some examples**

```{literalinclude} checkers/callandmessage_example.c
:language: objc
```

(core-dividezero)=

#### core.DivideZero (C, C++, ObjC)

> Check for division by zero.

```{literalinclude} checkers/dividezero_example.c
:language: c
```

(core-nonnullparamchecker)=

#### core.NonNullParamChecker (C, C++, ObjC)

Check for null pointers passed as arguments to a function whose arguments are references or marked with the 'nonnull' attribute.

```cpp
int f(int *p) __attribute__((nonnull));

void test(int *p) {
  if (!p)
    f(p); // warn
}
```

(core-nulldereference)=

#### core.NullDereference (C, C++, ObjC)

Check for dereferences of null pointers.

```objc
// C
void test(int *p) {
  if (p)
    return;

  int x = p[0]; // warn
}

// C
void test(int *p) {
  if (!p)
    *p = 0; // warn
}

// C++
class C {
public:
  int x;
};

void test() {
  C *pc = 0;
  int k = pc->x; // warn
}

// Objective-C
@interface MyClass {
@public
  int x;
}
@end

void test() {
  MyClass *obj = 0;
  obj->x = 1; // warn
}
```

Null pointer dereferences of pointers with address spaces are not always defined
as error. Specifically on x86/x86-64 target if the pointer address space is
256 (x86 GS Segment), 257 (x86 FS Segment), or 258 (x86 SS Segment), a null
dereference is not defined as error. See [X86/X86-64 Language Extensions](https://clang.llvm.org/docs/LanguageExtensions.html#memory-references-to-specified-segments)
for reference.

If the analyzer option `suppress-dereferences-from-any-address-space` is set
to true (the default value), then this checker never reports dereference of
pointers with a specified address space. If the option is set to false, then
reports from the specific x86 address spaces 256, 257 and 258 are still
suppressed, but null dereferences from other address spaces are reported.

(core-nullpointerarithm)=

#### core.NullPointerArithm (C, C++)

Check for undefined arithmetic operations with null pointers.

The checker can detect the following cases:

> - `p + x` and `x + p` where `p` is a null pointer and `x` is a nonzero
>   integer value.
> - `p - x` where `p` is a null pointer and `x` is a nonzero integer
>   value.
> - `p1 - p2` where one of `p1` and `p2` is null and the other a
>   non-null pointer.

Result of these operations is undefined according to the standard.
In the above listed cases, the checker will warn even if the expression
described to be "nonzero" or "non-null" has unknown value, because it is likely
that it can have non-zero value during the program execution.

```c
void test1(int *p, int offset) {
  if (p)
    return;

  int *p1 = p + offset; // warn: 'p' is null, 'offset' is unknown but likely non-zero
}

void test2(int *p, int offset) {
  if (p) { } // this indicates that it is possible for 'p' to be null
  if (offset == 0)
    return;

  int *p1 = p - offset; // warn: 'p' is null, 'offset' is known to be non-zero
}

void test3(char *p1, char *p2) {
  if (p1)
    return;

  int a = p1 - p2; // warn: 'p1' is null, 'p2' can be likely non-null
}
```

(core-stackaddressescape)=

#### core.StackAddressEscape (C)

Check that addresses to stack memory do not escape the function.

```c
char const *p;

void test() {
  char const str[] = "string";
  p = str; // warn
}

void* test() {
   return __builtin_alloca(12); // warn
}

void test() {
  static int *x;
  int y;
  x = &y; // warn
}
```

(core-undefinedbinaryoperatorresult)=

#### core.UndefinedBinaryOperatorResult (C)

Check for undefined results of binary operators.

```c
void test() {
  int x;
  int y = x + 1; // warn: left operand is garbage
}
```

(core-vlasize)=

#### core.VLASize (C)

Check for declarations of Variable Length Arrays (VLA) of undefined, zero or negative
size.

```c
void test() {
  int x;
  int vla1[x]; // warn: garbage as size
}

void test() {
  int x = 0;
  int vla2[x]; // warn: zero size
}
```

The checker also gives warning if the `TaintPropagation` checker is switched on
and an unbound, attacker controlled (tainted) value is used to define
the size of the VLA.

```c
void taintedVLA(void) {
  int x;
  scanf("%d", &x);
  int vla[x]; // Declared variable-length array (VLA) has tainted (attacker controlled) size, that can be 0 or negative
}

void taintedVerfieidVLA(void) {
  int x;
  scanf("%d", &x);
  if (x<1)
    return;
  int vla[x]; // no-warning. The analyzer can prove that x must be positive.
}
```

(core-uninitialized-arraysubscript)=

#### core.uninitialized.ArraySubscript (C)

Check for uninitialized values used as array subscripts.

```c
void test() {
  int i, a[10];
  int x = a[i]; // warn: array subscript is undefined
}
```

(core-uninitialized-assign)=

#### core.uninitialized.Assign (C)

Check for assigning uninitialized values.

```c
void test() {
  int x;
  x |= 1; // warn: left expression is uninitialized
}
```

(core-uninitialized-branch)=

#### core.uninitialized.Branch (C)

Check for uninitialized values used as branch conditions.

```c
void test() {
  int x;
  if (x) // warn
    return;
}
```

(core-uninitialized-capturedblockvariable)=

#### core.uninitialized.CapturedBlockVariable (C)

Check for blocks that capture uninitialized values.

```c
void test() {
  int x;
  ^{ int y = x; }(); // warn
}
```

(core-uninitialized-undefreturn)=

#### core.uninitialized.UndefReturn (C)

Check for uninitialized values being returned to the caller.

```c
int test() {
  int x;
  return x; // warn
}
```

(core-uninitialized-newarraysize)=

#### core.uninitialized.NewArraySize (C++)

Check if the element count in new[] is garbage or undefined.

```cpp
void test() {
  int n;
  int *arr = new int[n]; // warn: Element count in new[] is a garbage value
  delete[] arr;
}
```

(cplusplus-checkers)=

### cplusplus

C++ Checkers.

(cplusplus-arraydelete)=

#### cplusplus.ArrayDelete (C++)

Reports destructions of arrays of polymorphic objects that are destructed as
their base class. If the dynamic type of the array is different from its static
type, calling `delete[]` is undefined.

This checker corresponds to the SEI CERT rule [EXP51-CPP: Do not delete an array through a pointer of the incorrect type](https://wiki.sei.cmu.edu/confluence/display/cplusplus/EXP51-CPP.+Do+not+delete+an+array+through+a+pointer+of+the+incorrect+type).

```cpp
class Base {
public:
  virtual ~Base() {}
};
class Derived : public Base {};

Base *create() {
  Base *x = new Derived[10]; // note: Casting from 'Derived' to 'Base' here
  return x;
}

void foo() {
  Base *x = create();
  delete[] x; // warn: Deleting an array of 'Derived' objects as their base class 'Base' is undefined
}
```

**Limitations**

The checker does not emit note tags when casting to and from reference types,
even though the pointer values are tracked across references.

```cpp
void foo() {
  Derived *d = new Derived[10];
  Derived &dref = *d;

  Base &bref = static_cast<Base&>(dref); // no note
  Base *b = &bref;
  delete[] b; // warn: Deleting an array of 'Derived' objects as their base class 'Base' is undefined
}
```

(cplusplus-innerpointer)=

#### cplusplus.InnerPointer (C++)

Check for inner pointers of C++ containers used after re/deallocation.

Many container methods in the C++ standard library are known to invalidate
"references" (including actual references, iterators and raw pointers) to
elements of the container. Using such references after they are invalidated
causes undefined behavior, which is a common source of memory errors in C++ that
this checker is capable of finding.

The checker is currently limited to `std::string` objects and doesn't
recognize some of the more sophisticated approaches to passing unowned pointers
around, such as `std::string_view`.

```cpp
void deref_after_assignment() {
  std::string s = "llvm";
  const char *c = s.data(); // note: pointer to inner buffer of 'std::string' obtained here
  s = "clang"; // note: inner buffer of 'std::string' reallocated by call to 'operator='
  consume(c); // warn: inner pointer of container used after re/deallocation
}

const char *return_temp(int x) {
  return std::to_string(x).c_str(); // warn: inner pointer of container used after re/deallocation
  // note: pointer to inner buffer of 'std::string' obtained here
  // note: inner buffer of 'std::string' deallocated by call to destructor
}
```

(cplusplus-move)=

#### cplusplus.Move (C++)

Find use-after-move bugs in C++. This includes method calls on moved-from
objects, assignment of a moved-from object, and repeated move of a moved-from
object.

```cpp
struct A {
  void foo() {}
};

void f1() {
  A a;
  A b = std::move(a); // note: 'a' became 'moved-from' here
  a.foo();            // warn: method call on a 'moved-from' object 'a'
}

void f2() {
  A a;
  A b = std::move(a);
  A c(std::move(a)); // warn: move of an already moved-from object
}

void f3() {
  A a;
  A b = std::move(a);
  b = a; // warn: copy of moved-from object
}
```

The checker option `WarnOn` controls on what objects the use-after-move is
checked:

- The most strict value is `KnownsOnly`, in this mode only objects are
  checked whose type is known to be move-unsafe. These include most STL objects
  (but excluding move-safe ones) and smart pointers.
- With option value `KnownsAndLocals` local variables (of any type) are
  additionally checked. The idea behind this is that local variables are
  usually not tempting to be re-used so an use after move is more likely a bug
  than with member variables.
- With option value `All` any use-after move condition is checked on all
  kinds of variables, excluding global variables and known move-safe cases.

Default value is `KnownsAndLocals`.

Calls of methods named `empty()` or `isEmpty()` are allowed on moved-from
objects because these methods are considered as move-safe. Functions called
`reset()`, `destroy()`, `clear()`, `assign`, `resize`, `shrink` are
treated as state-reset functions and are allowed on moved-from objects, these
make the object valid again. This applies to any type of object (not only STL
ones).

(cplusplus-newdelete)=

#### cplusplus.NewDelete (C++)

Check for double-free and use-after-free problems. Traces memory managed by new/delete.

Custom allocation/deallocation functions can be defined using
{ref}`ownership attributes<analyzer-ownership-attrs>`.

```{literalinclude} checkers/newdelete_example.cpp
:language: cpp
```

(cplusplus-newdeleteleaks)=

#### cplusplus.NewDeleteLeaks (C++)

Check for memory leaks. Traces memory managed by new/delete.

Custom allocation/deallocation functions can be defined using
{ref}`ownership attributes<analyzer-ownership-attrs>`.

```cpp
void test() {
  int *p = new int;
} // warn
```

(cplusplus-placementnew)=

#### cplusplus.PlacementNew (C++)

Check if default placement new is provided with pointers to sufficient storage capacity.

```cpp
#include <new>

void f() {
  short s;
  long *lp = ::new (&s) long; // warn
}
```

(cplusplus-selfassignment)=

#### cplusplus.SelfAssignment (C++)

Checks C++ copy and move assignment operators for self assignment.

(cplusplus-stringchecker)=

#### cplusplus.StringChecker (C++)

Checks std::string operations.

Checks if the cstring pointer from which the `std::string` object is
constructed is `NULL` or not.
If the checker cannot reason about the nullness of the pointer it will assume
that it was non-null to satisfy the precondition of the constructor.

This checker is capable of checking the [SEI CERT C++ coding rule STR51-CPP.
Do not attempt to create a std::string from a null pointer](https://wiki.sei.cmu.edu/confluence/x/E3s-BQ).

```cpp
#include <string>

void f(const char *p) {
  if (!p) {
    std::string msg(p); // warn: The parameter must not be null
  }
}
```

(cplusplus-purevirtualcall)=

#### cplusplus.PureVirtualCall (C++)

When [virtual methods are called during construction and destruction](https://en.cppreference.com/w/cpp/language/virtual#During_construction_and_destruction)
the polymorphism is restricted to the class that's being constructed or
destructed because the more derived contexts are either not yet initialized or
already destructed.

This checker reports situations where this restricted polymorphism causes a
call to a pure virtual method, which is undefined behavior. (See also the
related checker {ref}`optin-cplusplus-VirtualCall` which reports situations
where the restricted polymorphism affects a call and the called method is not
pure virtual – but may be still surprising for the programmer.)

```cpp
struct A {
  virtual int getKind() = 0;

  A() {
    // warn: This calls the pure virtual method A::getKind().
    log << "Constructing " << getKind();
  }
  virtual ~A() {
    releaseResources();
  }
  void releaseResources() {
    // warn: This can call the pure virtual method A::getKind() when this is
    // called from the destructor.
    callSomeFunction(getKind());
  }
};
```

(deadcode-checkers)=

### deadcode

Dead Code Checkers.

(deadcode-deadstores)=

#### deadcode.DeadStores (C)

Check for values stored to variables that are never read afterwards.

```c
void test() {
  int x;
  x = 1; // warn
}
```

The `WarnForDeadNestedAssignments` option enables the checker to emit
warnings for nested dead assignments. You can disable with the
`-analyzer-config deadcode.DeadStores:WarnForDeadNestedAssignments=false`.
*Defaults to true*.

Would warn for this e.g.:
if ((y = make_int())) {
}

(nullability-checkers)=

### nullability

Checkers (mostly Objective C) that warn for null pointer passing and dereferencing errors.

(nullability-nullpassedtononnull)=

#### nullability.NullPassedToNonnull (ObjC)

Warns when a null pointer is passed to a pointer which has a `_Nonnull` type.

```objc
if (name != nil)
  return;
// Warning: nil passed to a callee that requires a non-null 1st parameter
NSString *greeting = [@"Hello " stringByAppendingString:name];
```

(nullability-nullreturnedfromnonnull)=

#### nullability.NullReturnedFromNonnull (C, C++, ObjC)

Warns when a null pointer is returned from a function that has `_Nonnull` return type.

```objc
- (nonnull id)firstChild {
  id result = nil;
  if ([_children count] > 0)
    result = _children[0];

  // Warning: nil returned from a method that is expected
  // to return a non-null value
  return result;
}
```

Warns when a null pointer is returned from a function annotated with `__attribute__((returns_nonnull))`

```cpp
int global;
__attribute__((returns_nonnull)) void* getPtr(void* p);

void* getPtr(void* p) {
  if (p) { // forgot to negate the condition
    return &global;
  }
  // Warning: nullptr returned from a function that is expected
  // to return a non-null value
  return p;
}
```

(nullability-nullabledereferenced)=

#### nullability.NullableDereferenced (ObjC)

Warns when a nullable pointer is dereferenced.

```objc
struct LinkedList {
  int data;
  struct LinkedList *next;
};

struct LinkedList * _Nullable getNext(struct LinkedList *l);

void updateNextData(struct LinkedList *list, int newData) {
  struct LinkedList *next = getNext(list);
  // Warning: Nullable pointer is dereferenced
  next->data = 7;
}
```

(nullability-nullablepassedtononnull)=

#### nullability.NullablePassedToNonnull (ObjC)

Warns when a nullable pointer is passed to a pointer which has a `_Nonnull` type.

```objc
typedef struct Dummy { int val; } Dummy;
Dummy *_Nullable returnsNullable();
void takesNonnull(Dummy *_Nonnull);

void test() {
  Dummy *p = returnsNullable();
  takesNonnull(p); // warn
}
```

(nullability-nullablereturnedfromnonnull)=

#### nullability.NullableReturnedFromNonnull (ObjC)

Warns when a nullable pointer is returned from a function that has `_Nonnull` return type.

(optin-checkers)=

### optin

Checkers for portability, performance, optional security and coding style specific rules.

(optin-core-enumcastoutofrange)=

#### optin.core.EnumCastOutOfRange (C, C++)

Check for integer to enumeration casts that would produce a value with no
corresponding enumerator. This is not necessarily undefined behavior, but can
lead to nasty surprises, so projects may decide to use a coding standard that
disallows these "unusual" conversions.

Note that no warnings are produced when the enum type (e.g. `std::byte`) has no
enumerators at all.

```cpp
enum WidgetKind { A=1, B, C, X=99 };

void foo() {
  WidgetKind c = static_cast<WidgetKind>(3);  // OK
  WidgetKind x = static_cast<WidgetKind>(99); // OK
  WidgetKind d = static_cast<WidgetKind>(4);  // warn
}
```

**Limitations**

This checker does not accept the coding pattern where an enum type is used to
store combinations of flag values.
Such enums should be annotated with the `__attribute__((flag_enum))` or by the
`[[clang::flag_enum]]` attribute to signal this intent. Refer to the
[documentation](https://clang.llvm.org/docs/AttributeReference.html#flag-enum)
of this Clang attribute.

```cpp
enum AnimalFlags
{
    HasClaws   = 1,
    CanFly     = 2,
    EatsFish   = 4,
    Endangered = 8
};

AnimalFlags operator|(AnimalFlags a, AnimalFlags b)
{
    return static_cast<AnimalFlags>(static_cast<int>(a) | static_cast<int>(b));
}

auto flags = HasClaws | CanFly;
```

Projects that use this pattern should not enable this optin checker.

(optin-core-fixedaddressdereference)=

#### optin.core.FixedAddressDereference (C, C++, ObjC)

Check for dereferences of fixed addresses.

A pointer contains a fixed address if it was set to a hard-coded value or it
becomes otherwise obvious that at that point it can have only a single fixed
numerical value.

```c
void test1() {
  int *p = (int *)0x020;
  int x = p[0]; // warn
}

void test2(int *p) {
  if (p == (int *)-1)
    *p = 0; // warn
}

void test3() {
  int (*p_function)(char, char);
  p_function = (int (*)(char, char))0x04080;
  int x = (*p_function)('x', 'y'); // NO warning yet at functon pointer calls
}
```

Access of fixed numerical addresses can be legitimate in low-level projects (e.g. firmware) and hardware interop.
These values are often marked as `volatile` (to prevent unwanted compiler optimizations),
so this checker doesn't report situations where the pointee of the fixed address is `volatile`.
If this suppression is not sufficient on a low-level project, then consider disabling this `optin` checker.
Note that null pointers will still be reported by {ref}`core.NullDereference <core-NullDereference>`
regardless if the pointee is `volatile` or not.

```c
void volatile_pointee() {
  *(volatile int *)0x404 = 1; // no warning: fixed non-null "volatile" pointee, you must know what you are doing
}

void deref_volatile_nullptr() {
  *(volatile int *)0 = 1; // core.NullDereference still warns about this
}
```

If the analyzer option `suppress-dereferences-from-any-address-space` is set
to true (the default value), then this checker never reports dereference of
pointers with a specified address space. If the option is set to false, then
reports from the specific x86 address spaces 256, 257 and 258 are still
suppressed, but fixed address dereferences from other address spaces are
reported.
Do not use the {ref}`address_space <langext-address_space_documentation>`
attribute to suppress the reports - it just happens so that the checker also doesn't raise issues if the attribute is present.

(optin-core-unconditionalvaarg)=

#### optin.core.UnconditionalVAArg (C, C++)

Check for variadic functions that unconditionally use `va_arg()`. It is
possible to use such functions safely (as long as they always receive at least
one variadic argument), but their careless use can lead to undefined behavior
(trying to access a variadic argument when there isn't any), so it is better to
avoid them.

**Note:** This checker only inspects the *definition* of the variadic functions
and reports those that *would* fail if they were called with no variadic
arguments -- even if there is no such call in the codebase.

This design rule is dictated by the SEI CERT rule [EXP47-C](https://wiki.sei.cmu.edu/confluence/display/c/EXP47-C.+Do+not+call+va_arg+with+an+argument+of+the+incorrect+type),
which describes several issues related to the use of `va_arg()`. (The problem
reported by this checker is shown in the second code example; the first,
unrelated code example is covered by the clang diagnostic [-Wvarargs](https://clang.llvm.org/docs/DiagnosticsReference.html#wvarargs).)

```cpp
// This function expects a list of variadic arguments terminated by a NULL pointer.
void log_message(const char *msg, ...) {
  va_list va;
  const char *arg;
  printf("%s\n", msg);
  va_start(va, msg);
  while ((arg = va_arg(va, const char *))) {
    // warn: calls to 'log_message' always reach this va_arg() expression
    printf(" * %s\n", arg);
  }
  va_end(va);
}
```

Instead of this bugprone pattern, the SEI-CERT coding standard suggests
approaches where the number of variadic arguments can be specified in a
different manner -- for example, it can be encoded in the initial parameter
like `void foo(size_t num_varargs, ...)`. Those approaches are still not
foolproof (e.g. `foo(3, "a")` will be undefined behavior), but they reduce
the chances of an accidental mistake.

(optin-cplusplus-uninitializedobject)=

#### optin.cplusplus.UninitializedObject (C++)

This checker reports uninitialized fields in objects created after a constructor
call. It doesn't only find direct uninitialized fields, but rather makes a deep
inspection of the object, analyzing all of its fields' subfields.
The checker regards inherited fields as direct fields, so one will receive
warnings for uninitialized inherited data members as well.

```cpp
// With Pedantic and CheckPointeeInitialization set to true

struct A {
  struct B {
    int x; // note: uninitialized field 'this->b.x'
    // note: uninitialized field 'this->bptr->x'
    int y; // note: uninitialized field 'this->b.y'
    // note: uninitialized field 'this->bptr->y'
  };
  int *iptr; // note: uninitialized pointer 'this->iptr'
  B b;
  B *bptr;
  char *cptr; // note: uninitialized pointee 'this->cptr'

  A (B *bptr, char *cptr) : bptr(bptr), cptr(cptr) {}
};

void f() {
  A::B b;
  char c;
  A a(&b, &c); // warning: 6 uninitialized fields
 //          after the constructor call
}

// With Pedantic set to false and
// CheckPointeeInitialization set to true
// (every field is uninitialized)

struct A {
  struct B {
    int x;
    int y;
  };
  int *iptr;
  B b;
  B *bptr;
  char *cptr;

  A (B *bptr, char *cptr) : bptr(bptr), cptr(cptr) {}
};

void f() {
  A::B b;
  char c;
  A a(&b, &c); // no warning
}

// With Pedantic set to true and
// CheckPointeeInitialization set to false
// (pointees are regarded as initialized)

struct A {
  struct B {
    int x; // note: uninitialized field 'this->b.x'
    int y; // note: uninitialized field 'this->b.y'
  };
  int *iptr; // note: uninitialized pointer 'this->iptr'
  B b;
  B *bptr;
  char *cptr;

  A (B *bptr, char *cptr) : bptr(bptr), cptr(cptr) {}
};

void f() {
  A::B b;
  char c;
  A a(&b, &c); // warning: 3 uninitialized fields
 //          after the constructor call
}
```

**Options**

This checker has several options which can be set from command line (e.g.
`-analyzer-config optin.cplusplus.UninitializedObject:Pedantic=true`):

- `Pedantic` (boolean). If to false, the checker won't emit warnings for
  objects that don't have at least one initialized field. Defaults to false.
- `NotesAsWarnings` (boolean). If set to true, the checker will emit a
  warning for each uninitialized field, as opposed to emitting one warning per
  constructor call, and listing the uninitialized fields that belongs to it in
  notes. *Defaults to false*.
- `CheckPointeeInitialization` (boolean). If set to false, the checker will
  not analyze the pointee of pointer/reference fields, and will only check
  whether the object itself is initialized. *Defaults to false*.
- `IgnoreRecordsWithField` (string). If supplied, the checker will not analyze
  structures that have a field with a name or type name that matches the given
  pattern. *Defaults to ""*.

(optin-cplusplus-virtualcall)=

#### optin.cplusplus.VirtualCall (C++)

When [virtual methods are called during construction and destruction](https://en.cppreference.com/w/cpp/language/virtual#During_construction_and_destruction)
the polymorphism is restricted to the class that's being constructed or
destructed because the more derived contexts are either not yet initialized or
already destructed.

Although this behavior is well-defined, it can surprise the programmer and
cause unintended behavior, so this checker reports calls that appear to be
virtual calls but can be affected by this restricted polymorphism.

Note that situations where this restricted polymorphism causes a call to a pure
virtual method (which is definitely invalid, triggers undefined behavior) are
**reported by another checker:** {ref}`cplusplus-PureVirtualCall` and **this
checker does not report them**.

```cpp
struct A {
  virtual int getKind();

  A() {
    // warn: This calls A::getKind() even if we are constructing an instance
    // of a different class that is derived from A.
    log << "Constructing " << getKind();
  }
  virtual ~A() {
    releaseResources();
  }
  void releaseResources() {
    // warn: This can be called within ~A() and calls A::getKind() even if
    // we are destructing a class that is derived from A.
    callSomeFunction(getKind());
  }
};
```

(optin-mpi-mpi-checker)=

#### optin.mpi.MPI-Checker (C)

Checks MPI code.

```c
void test() {
  double buf = 0;
  MPI_Request sendReq1;
  MPI_Ireduce(MPI_IN_PLACE, &buf, 1, MPI_DOUBLE, MPI_SUM,
      0, MPI_COMM_WORLD, &sendReq1);
} // warn: request 'sendReq1' has no matching wait.

void test() {
  double buf = 0;
  MPI_Request sendReq;
  MPI_Isend(&buf, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &sendReq);
  MPI_Irecv(&buf, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &sendReq); // warn
  MPI_Isend(&buf, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &sendReq); // warn
  MPI_Wait(&sendReq, MPI_STATUS_IGNORE);
}

void missingNonBlocking() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Request sendReq1[10][10][10];
  MPI_Wait(&sendReq1[1][7][9], MPI_STATUS_IGNORE); // warn
}
```

(optin-osx-cocoa-localizability-emptylocalizationcontextchecker)=

#### optin.osx.cocoa.localizability.EmptyLocalizationContextChecker (ObjC)

Check that NSLocalizedString macros include a comment for context.

```objc
- (void)test {
  NSString *string = NSLocalizedString(@"LocalizedString", nil); // warn
  NSString *string2 = NSLocalizedString(@"LocalizedString", @" "); // warn
  NSString *string3 = NSLocalizedStringWithDefaultValue(
    @"LocalizedString", nil, [[NSBundle alloc] init], nil,@""); // warn
}
```

(optin-osx-cocoa-localizability-nonlocalizedstringchecker)=

#### optin.osx.cocoa.localizability.NonLocalizedStringChecker (ObjC)

Warns about uses of non-localized NSStrings passed to UI methods expecting localized NSStrings.

```objc
NSString *alarmText =
  NSLocalizedString(@"Enabled", @"Indicates alarm is turned on");
if (!isEnabled) {
  alarmText = @"Disabled";
}
UILabel *alarmStateLabel = [[UILabel alloc] init];

// Warning: User-facing text should use localized string macro
[alarmStateLabel setText:alarmText];
```

(optin-performance-gcdantipattern)=

#### optin.performance.GCDAntipattern

Check for performance anti-patterns when using Grand Central Dispatch.

(optin-performance-padding)=

#### optin.performance.Padding (C, C++, ObjC)

Check for excessively padded structs.

This checker detects structs with excessive padding, which can lead to wasted
memory thus decreased performance by reducing the effectiveness of the
processor cache. Padding bytes are added by compilers to align data accesses
as some processors require data to be aligned to certain boundaries. On others,
unaligned data access are possible, but impose significantly larger latencies.

To avoid padding bytes, the fields of a struct should be ordered by decreasing
by alignment. Usually, its easier to think of the `sizeof` of the fields, and
ordering the fields by `sizeof` would usually also lead to the same optimal
layout.

In rare cases, one can use the `#pragma pack(1)` directive to enforce a packed
layout too, but it can significantly increase the access times, so reordering the
fields is usually a better solution.

```cpp
// warn: Excessive padding in 'struct NonOptimal' (35 padding bytes, where 3 is optimal)
struct NonOptimal {
  char c1;
  // 7 bytes of padding
  std::int64_t big1; // 8 bytes
  char c2;
  // 7 bytes of padding
  std::int64_t big2; // 8 bytes
  char c3;
  // 7 bytes of padding
  std::int64_t big3; // 8 bytes
  char c4;
  // 7 bytes of padding
  std::int64_t big4; // 8 bytes
  char c5;
  // 7 bytes of padding
};
static_assert(sizeof(NonOptimal) == 4*8+5+5*7);

// no-warning: The fields are nicely aligned to have the minimal amount of padding bytes.
struct Optimal {
  std::int64_t big1; // 8 bytes
  std::int64_t big2; // 8 bytes
  std::int64_t big3; // 8 bytes
  std::int64_t big4; // 8 bytes
  char c1;
  char c2;
  char c3;
  char c4;
  char c5;
  // 3 bytes of padding
};
static_assert(sizeof(Optimal) == 4*8+5+3);

// no-warning: Bit packing representation is also accepted by this checker, but
// it can significantly increase access times, so prefer reordering the fields.
#pragma pack(1)
struct BitPacked {
  char c1;
  std::int64_t big1; // 8 bytes
  char c2;
  std::int64_t big2; // 8 bytes
  char c3;
  std::int64_t big3; // 8 bytes
  char c4;
  std::int64_t big4; // 8 bytes
  char c5;
};
static_assert(sizeof(BitPacked) == 4*8+5);
```

The `AllowedPad` option can be used to specify a threshold for the number
padding bytes raising the warning. If the number of padding bytes of the struct
and the optimal number of padding bytes differ by more than the threshold value,
a warning will be raised.

By default, the `AllowedPad` threshold is 24 bytes.

To override this threshold to e.g. 4 bytes, use the
`-analyzer-config optin.performance.Padding:AllowedPad=4` option.

(optin-portability-unixapi)=

#### optin.portability.UnixAPI

Reports situations where 0 is passed as the "size" argument of various
allocation functions ( `calloc`, `malloc`, `realloc`, `reallocf`,
`alloca`, `__builtin_alloca`, `__builtin_alloca_with_align`, `valloc`).

Note that similar functionality is also supported by {ref}`unix-Malloc` which
reports code that *uses* memory allocated with size zero.

(The name of this checker is motivated by the fact that it was originally
introduced with the vague goal that it "Finds implementation-defined behavior
in UNIX/Posix functions.")

### optin.taint

Checkers implementing
[taint analysis](https://en.wikipedia.org/wiki/Taint_checking).

(optin-taint-generictaint)=

#### optin.taint.GenericTaint (C, C++)

Taint analysis identifies potential security vulnerabilities where the
attacker can inject malicious data to the program to execute an attack
(privilege escalation, command injection, SQL injection etc.).

The malicious data is injected at the taint source (e.g. `getenv()` call)
which is then propagated through function calls and being used as arguments of
sensitive operations, also called as taint sinks (e.g. `system()` call).

One can defend against this type of vulnerability by always checking and
sanitizing the potentially malicious, untrusted user input.

The goal of the checker is to discover and show to the user these potential
taint source-sink pairs and the propagation call chain.

The most notable examples of taint sources are:

> - data from network
> - files or standard input
> - environment variables
> - data from databases

Let us examine a practical example of a Command Injection attack.

```c
// Command Injection Vulnerability Example
int main(int argc, char** argv) {
  char cmd[2048] = "/bin/cat ";
  char filename[1024];
  printf("Filename:");
  scanf (" %1023[^\n]", filename); // The attacker can inject a shell escape here
  strcat(cmd, filename);
  system(cmd); // Warning: Untrusted data is passed to a system call
}
```

The program prints the content of any user specified file.
Unfortunately the attacker can execute arbitrary commands
with shell escapes. For example with the following input the `ls` command is also
executed after the contents of `/etc/shadow` is printed.
`Input: /etc/shadow ; ls /`

The analysis implemented in this checker points out this problem.

One can protect against such attack by for example checking if the provided
input refers to a valid file and removing any invalid user input.

```c
// No vulnerability anymore, but we still get the warning
void sanitizeFileName(char* filename){
  if (access(filename,F_OK)){// Verifying user input
    printf("File does not exist\n");
    filename[0]='\0';
    }
}
int main(int argc, char** argv) {
  char cmd[2048] = "/bin/cat ";
  char filename[1024];
  printf("Filename:");
  scanf (" %1023[^\n]", filename); // The attacker can inject a shell escape here
  sanitizeFileName(filename);// filename is safe after this point
  if (!filename[0])
    return -1;
  strcat(cmd, filename);
  system(cmd); // Superfluous Warning: Untrusted data is passed to a system call
}
```

Unfortunately, the checker cannot discover automatically that the programmer
have performed data sanitation, so it still emits the warning.

One can get rid of this superfluous warning by telling by specifying the
sanitation functions in the taint configuration file (see
{doc}`user-docs/TaintAnalysisConfiguration`).

```yaml
Filters:
- Name: sanitizeFileName
  Args: [0]
```

The clang invocation to pass the configuration file location:

```bash
clang  --analyze -Xclang -analyzer-config  -Xclang optin.taint.TaintPropagation:Config=`pwd`/taint_config.yml ...
```

If you are validating your inputs instead of sanitizing them, or don't want to
mention each sanitizing function in our configuration,
you can use a more generic approach.

Introduce a generic no-op `csa_mark_sanitized(..)` function to
tell the Clang Static Analyzer
that the variable is safe to be used on that analysis path.

```c
// Marking sanitized variables safe.
// No vulnerability anymore, no warning.

// User csa_mark_sanitize function is for the analyzer only
#ifdef __clang_analyzer__
  void csa_mark_sanitized(const void *);
#endif

int main(int argc, char** argv) {
  char cmd[2048] = "/bin/cat ";
  char filename[1024];
  printf("Filename:");
  scanf (" %1023[^\n]", filename);
  if (access(filename,F_OK)){// Verifying user input
    printf("File does not exist\n");
    return -1;
  }
  #ifdef __clang_analyzer__
    csa_mark_sanitized(filename); // Indicating to CSA that filename variable is safe to be used after this point
  #endif
  strcat(cmd, filename);
  system(cmd); // No warning
}
```

Similarly to the previous example, you need to
define a `Filter` function in a `YAML` configuration file
and add the `csa_mark_sanitized` function.

```yaml
Filters:
- Name: csa_mark_sanitized
  Args: [0]
```

Then calling `csa_mark_sanitized(X)` will tell the analyzer that `X` is safe to
be used after this point, because its contents are verified. It is the
responsibility of the programmer to ensure that this verification was indeed
correct. Please note that `csa_mark_sanitized` function is only declared and
used during Clang Static Analysis and skipped in (production) builds.

Further examples of injection vulnerabilities this checker can find.

```c
void test() {
  char x = getchar(); // 'x' marked as tainted
  system(&x); // warn: untrusted data is passed to a system call
}

// note: compiler internally checks if the second param to
// sprintf is a string literal or not.
// Use -Wno-format-security to suppress compiler warning.
void test() {
  char s[10], buf[10];
  fscanf(stdin, "%s", s); // 's' marked as tainted

  sprintf(buf, s); // warn: untrusted data used as a format string
}
```

There are built-in sources, propagations and sinks even if no external taint
configuration is provided.

Default sources:

: `_IO_getc`, `fdopen`, `fopen`, `freopen`, `get_current_dir_name`,
  `getch`, `getchar`, `getchar_unlocked`, `getwd`, `getcwd`,
  `getgroups`, `gethostname`, `getlogin`, `getlogin_r`, `getnameinfo`,
  `gets`, `gets_s`, `getseuserbyname`, `readlink`, `readlinkat`,
  `scanf`, `scanf_s`, `socket`, `wgetch`

Default propagations rules:

: `atoi`, `atol`, `atoll`, `basename`, `dirname`, `fgetc`,
  `fgetln`, `fgets`, `fnmatch`, `fread`, `fscanf`, `fscanf_s`,
  `index`, `inflate`, `isalnum`, `isalpha`, `isascii`, `isblank`,
  `iscntrl`, `isdigit`, `isgraph`, `islower`, `isprint`, `ispunct`,
  `isspace`, `isupper`, `isxdigit`, `memchr`, `memrchr`, `sscanf`,
  `getc`, `getc_unlocked`, `getdelim`, `getline`, `getw`, `memcmp`,
  `memcpy`, `memmem`, `memmove`, `mbtowc`, `pread`, `qsort`,
  `qsort_r`, `rawmemchr`, `read`, `recv`, `recvfrom`, `rindex`,
  `strcasestr`, `strchr`, `strchrnul`, `strcasecmp`, `strcmp`,
  `strcspn`, `strncasecmp`, `strncmp`, `strndup`,
  `strndupa`, `strpbrk`, `strrchr`, `strsep`, `strspn`,
  `strstr`, `strtol`, `strtoll`, `strtoul`, `strtoull`, `tolower`,
  `toupper`, `ttyname`, `ttyname_r`, `wctomb`, `wcwidth`

Default sinks:

: `printf`, `setproctitle`, `system`, `popen`, `execl`, `execle`,
  `execlp`, `execv`, `execvp`, `execvP`, `execve`, `dlopen`

Please note that there are no built-in filter functions.

One can configure their own taint sources, sinks, and propagation rules by
providing a configuration file via checker option
`optin.taint.TaintPropagation:Config`. The configuration file is in
[YAML](http://llvm.org/docs/YamlIO.html#introduction-to-yaml) format. The
taint-related options defined in the config file extend but do not override the
built-in sources, rules, sinks. The format of the external taint configuration
file is not stable, and could change without any notice even in a non-backward
compatible way.

For a more detailed description of configuration options, please see the
{doc}`user-docs/TaintAnalysisConfiguration`. For an example see
{ref}`clangsa-taint-configuration-example`.

**Configuration**

- `optin.taint.TaintPropagation:Config` Specifies the name of the YAML
  configuration file. The user can define their own taint sources and sinks.
- `optin.taint.TaintPropagation:EnableDefaultConfig` If set to false,
  : the default source, sink and propagation rules are not loaded. This way,
    advanced users can fully customize their taint configuration model.
    Default: `true`.
- If the analyzer option `assume-controlled-environment` is set to `false`,
  it is assumed that the command line arguments and the environment
  variables of the program are attacker controlled.
  In particular, the `argv`, `argc` and `envp` arguments of the
  `main` function and the return value of the `getenv()`
  function are assumed to hold tainted values.

**Related Guidelines**

- [CWE Data Neutralization Issues](https://cwe.mitre.org/data/definitions/137.html)
- [SEI Cert STR02-C. Sanitize data passed to complex subsystems](https://wiki.sei.cmu.edu/confluence/display/c/STR02-C.+Sanitize+data+passed+to+complex+subsystems)
- [SEI Cert ENV33-C. Do not call system()](https://wiki.sei.cmu.edu/confluence/pages/viewpage.action?pageId=87152177)
- [ENV03-C. Sanitize the environment when invoking external programs](https://wiki.sei.cmu.edu/confluence/display/c/ENV03-C.+Sanitize+the+environment+when+invoking+external+programs)

**Limitations**

- The taintedness property is not propagated through function calls which are
  unknown (or too complex) to the analyzer, unless there is a specific
  propagation rule built-in to the checker or given in the YAML configuration
  file. This causes potential true positive findings to be lost.

(optin-taint-taintedalloc)=

#### optin.taint.TaintedAlloc (C, C++)

This checker warns for cases when the `size` parameter of the `malloc` ,
`calloc`, `realloc`, `alloca` or the size parameter of the
array new C++ operator is tainted (potentially attacker controlled).
If an attacker can inject a large value as the size parameter, memory exhaustion
denial of service attack can be carried out.

The analyzer emits warning only if it cannot prove that the size parameter is
within reasonable bounds (`<= SIZE_MAX/4`). This functionality partially
covers the SEI Cert coding standard rule [INT04-C](https://wiki.sei.cmu.edu/confluence/display/c/INT04-C.+Enforce+limits+on+integer+values+originating+from+tainted+sources).

You can silence this warning either by bound checking the `size` parameter, or
by explicitly marking the `size` parameter as sanitized. See the
{ref}`optin-taint-GenericTaint` checker for an example.

Custom allocation/deallocation functions can be defined using
{ref}`ownership attributes<analyzer-ownership-attrs>`.

```c
void vulnerable(void) {
  size_t size = 0;
  scanf("%zu", &size);
  int *p = malloc(size); // warn: malloc is called with a tainted (potentially attacker controlled) value
  free(p);
}

void not_vulnerable(void) {
  size_t size = 0;
  scanf("%zu", &size);
  if (1024 < size)
    return;
  int *p = malloc(size); // No warning expected as the the user input is bound
  free(p);
}

void vulnerable_cpp(void) {
  size_t size = 0;
  scanf("%zu", &size);
  int *ptr = new int[size];// warn: Memory allocation function is called with a tainted (potentially attacker controlled) value
  delete[] ptr;
}
```

(optin-taint-tainteddiv)=

#### optin.taint.TaintedDiv (C, C++, ObjC)

This checker warns when the denominator in a division
operation is a tainted (potentially attacker controlled) value.
If the attacker can set the denominator to 0, a runtime error can
be triggered. The checker warns when the denominator is a tainted
value and the analyzer cannot prove that it is not 0. This warning
is more pessimistic than the {ref}`core-DivideZero` checker
which warns only when it can prove that the denominator is 0.

```c
int vulnerable(int n) {
  size_t size = 0;
  scanf("%zu", &size);
  return n / size; // warn: Division by a tainted value, possibly zero
}

int not_vulnerable(int n) {
  size_t size = 0;
  scanf("%zu", &size);
  if (!size)
    return 0;
  return n / size; // no warning
}
```

(security-checkers)=

### security

Security related checkers.

(security-arraybound)=

#### security.ArrayBound (C, C++)

Report out of bounds access to memory that is before the start or after the end
of the accessed region (array, heap-allocated region, string literal etc.).
This usually means incorrect indexing, but the checker also detects access via
the operators `*` and `->`.

```c
void test_underflow(int x) {
  int buf[100][100];
  if (x < 0)
    buf[0][x] = 1; // warn
}

void test_overflow() {
  int buf[100];
  int *p = buf + 100;
  *p = 1; // warn
}
```

If checkers like {ref}`unix-Malloc` or {ref}`cplusplus-NewDelete` are enabled
to model the behavior of `malloc()`, `operator new` and similar
allocators), then this checker can also reports out of bounds access to
dynamically allocated memory:

```cpp
int *test_dynamic() {
  int *mem = new int[100];
  mem[-1] = 42; // warn
  return mem;
}
```

In uncertain situations (when the checker can neither prove nor disprove that
overflow occurs), the checker assumes that the the index (more precisely, the
memory offeset) is within bounds.

However, if {ref}`optin-taint-GenericTaint` is enabled and the index/offset is
tainted (i.e. it is influenced by an untrusted source), then this checker
reports the potential out of bounds access:

```c
void test_with_tainted_index() {
  char s[] = "abc";
  int x = getchar();
  char c = s[x]; // warn: potential out of bounds access with tainted index
}
```

:::{note}
This checker is an improved and renamed version of the checker that was
previously known as `alpha.security.ArrayBoundV2`. The old checker
`alpha.security.ArrayBound` was removed when the (previously
"experimental") V2 variant became stable enough for regular use.
:::

(security-cert-env-invalidptr)=

#### security.cert.env.InvalidPtr

Corresponds to SEI CERT Rules [ENV31-C](https://wiki.sei.cmu.edu/confluence/display/c/ENV31-C.+Do+not+rely+on+an+environment+pointer+following+an+operation+that+may+invalidate+it) and [ENV34-C](https://wiki.sei.cmu.edu/confluence/display/c/ENV34-C.+Do+not+store+pointers+returned+by+certain+functions).

- **ENV31-C**:
  Rule is about the possible problem with `main` function's third argument, environment pointer,
  "envp". When environment array is modified using some modification function
  such as `putenv`, `setenv` or others, It may happen that memory is reallocated,
  however "envp" is not updated to reflect the changes and points to old memory
  region.
- **ENV34-C**:
  Some functions return a pointer to a statically allocated buffer.
  Consequently, subsequent call of these functions will invalidate previous
  pointer. These functions include: `getenv`, `localeconv`, `asctime`, `setlocale`, `strerror`

```c
int main(int argc, const char *argv[], const char *envp[]) {
  if (setenv("MY_NEW_VAR", "new_value", 1) != 0) {
    // setenv call may invalidate 'envp'
    /* Handle error */
  }
  if (envp != NULL) {
    for (size_t i = 0; envp[i] != NULL; ++i) {
      puts(envp[i]);
      // envp may no longer point to the current environment
      // this program has unanticipated behavior, since envp
      // does not reflect changes made by setenv function.
    }
  }
  return 0;
}

void previous_call_invalidation() {
  char *p, *pp;

  p = getenv("VAR");
  setenv("SOMEVAR", "VALUE", /*overwrite = */1);
  // call to 'setenv' may invalidate p

  *p;
  // dereferencing invalid pointer
}
```

The `InvalidatingGetEnv` option is available for treating `getenv` calls as
invalidating. When enabled, the checker issues a warning if `getenv` is called
multiple times and their results are used without first creating a copy.
This level of strictness might be considered overly pedantic for the commonly
used `getenv` implementations.

To enable this option, use:
`-analyzer-config security.cert.env.InvalidPtr:InvalidatingGetEnv=true`.

By default, this option is set to *false*.

When this option is enabled, warnings will be generated for scenarios like the
following:

```c
char* p = getenv("VAR");
char* pp = getenv("VAR2"); // assumes this call can invalidate `env`
strlen(p); // warns about accessing invalid ptr
```

(security-floatloopcounter)=

#### security.FloatLoopCounter (C)

Warn on using a floating point value as a loop counter (CERT: FLP30-C, FLP30-CPP).

```c
void test() {
  for (float x = 0.1f; x <= 1.0f; x += 0.1f) {} // warn
}
```

(security-insecureapi-uncheckedreturn)=

#### security.insecureAPI.UncheckedReturn (C)

Warn on uses of functions whose return values must be always checked.

```c
void test() {
  setuid(1); // warn
}
```

(security-insecureapi-bcmp)=

#### security.insecureAPI.bcmp (C)

Warn on uses of the 'bcmp' function.

```c
void test() {
  bcmp(ptr0, ptr1, n); // warn
}
```

(security-insecureapi-bcopy)=

#### security.insecureAPI.bcopy (C)

Warn on uses of the 'bcopy' function.

```c
void test() {
  bcopy(src, dst, n); // warn
}
```

(security-insecureapi-bzero)=

#### security.insecureAPI.bzero (C)

Warn on uses of the 'bzero' function.

```c
void test() {
  bzero(ptr, n); // warn
}
```

(security-insecureapi-decodevalueofobjctype)=

#### security.insecureAPI.decodeValueOfObjCType (C)

Warn on uses of the Objective-C method `-decodeValueOfObjCType:at:`.

```objc
void test(NSCoder *decoder) {
  unsigned int x;
  [decoder decodeValueOfObjCType:"I" at:&x]; // warn
}
```

This diagnostic is emitted only on Apple platforms where the safer
`-decodeValueOfObjCType:at:size:` alternative is available
(iOS 11+, macOS 10.13+, tvOS 11+, watchOS 4.0+).

(security-insecureapi-getpw)=

#### security.insecureAPI.getpw (C)

Warn on uses of the 'getpw' function.

```c
void test() {
  char buff[1024];
  getpw(2, buff); // warn
}
```

(security-insecureapi-gets)=

#### security.insecureAPI.gets (C)

Warn on uses of the 'gets' function.

```c
void test() {
  char buff[1024];
  gets(buff); // warn
}
```

(security-insecureapi-mkstemp)=

#### security.insecureAPI.mkstemp (C)

Warn when 'mkstemp' is passed fewer than 6 X's in the format string.

```c
void test() {
  mkstemp("XX"); // warn
}
```

(security-insecureapi-mktemp)=

#### security.insecureAPI.mktemp (C)

Warn on uses of the `mktemp` function.

```c
void test() {
  char *x = mktemp("/tmp/zxcv"); // warn: insecure, use mkstemp
}
```

(security-insecureapi-rand)=

#### security.insecureAPI.rand (C)

Warn on uses of inferior random number generating functions (only if arc4random function is available):
`drand48, erand48, jrand48, lcong48, lrand48, mrand48, nrand48, random, rand_r`.

```c
void test() {
  random(); // warn
}
```

(security-insecureapi-strcpy)=

#### security.insecureAPI.strcpy (C)

Warn on uses of the `strcpy` and `strcat` functions.

```c
void test() {
  char x[4];
  char *y = "abcd";

  strcpy(x, y); // warn
}
```

(security-insecureapi-vfork)=

#### security.insecureAPI.vfork (C)

> Warn on uses of the 'vfork' function.

```c
void test() {
  vfork(); // warn
}
```

(security-insecureapi-deprecatedorunsafebufferhandling)=

#### security.insecureAPI.DeprecatedOrUnsafeBufferHandling (C)

> Warn on occurrences of unsafe or deprecated buffer handling functions, which now have a secure variant: `sprintf, fprintf, vsprintf, scanf, wscanf, fscanf, fwscanf, vscanf, vwscanf, vfscanf, vfwscanf, sscanf, swscanf, vsscanf, vswscanf, swprintf, snprintf, vswprintf, vsnprintf, memcpy, memmove, strncpy, strncat, memset`

```c
void test() {
  char buf [5];
  strncpy(buf, "a", 1); // warn
}
```

The `ReportMode` option controls when warnings are reported:

- `all`: Reports all unsafe functions regardless of C standard or Annex K availability. Useful for security auditing and vulnerability scanning.
- `actionable`: Only reports when Annex K is available (C11 with `__STDC_LIB_EXT1__` and `__STDC_WANT_LIB_EXT1__=1`).
- `c11-only`: Reports when C11 standard is enabled (does not take Annex K availability into account).

To set this option, use:
`-analyzer-config security.insecureAPI.DeprecatedOrUnsafeBufferHandling:ReportMode=all`

By default, this option is set to *c11-only*.

(security-mmapwriteexec)=

#### security.MmapWriteExec (C)

Warn on `mmap()` calls with both writable and executable access.

```c
void test(int n) {
  void *c = mmap(NULL, 32, PROT_READ | PROT_WRITE | PROT_EXEC,
                 MAP_PRIVATE | MAP_ANON, -1, 0);
  // warn: Both PROT_WRITE and PROT_EXEC flags are set. This can lead to
  //       exploitable memory regions, which could be overwritten with malicious
  //       code
}
```

(security-pointersub)=

#### security.PointerSub (C)

Check for pointer subtractions on two pointers pointing to different memory
chunks. According to the C standard §6.5.6 only subtraction of pointers that
point into (or one past the end) the same array object is valid (for this
purpose non-array variables are like arrays of size 1). This checker only
searches for different memory objects at subtraction, but does not check if the
array index is correct. Furthermore, only cases are reported where
stack-allocated objects are involved (no warnings on pointers to memory
allocated by `malloc`).

```c
void test() {
  int a, b, c[10], d[10];
  int x = &c[3] - &c[1];
  x = &d[4] - &c[1]; // warn: 'c' and 'd' are different arrays
  x = (&a + 1) - &a;
  x = &b - &a; // warn: 'a' and 'b' are different variables
}

struct S {
  int x[10];
  int y[10];
};

void test1() {
  struct S a[10];
  struct S b;
  int d = &a[4] - &a[6];
  d = &a[0].x[3] - &a[0].x[1];
  d = a[0].y - a[0].x; // warn: 'S.b' and 'S.a' are different objects
  d = (char *)&b.y - (char *)&b.x; // warn: different members of the same object
  d = (char *)&b.y - (char *)&b; // warn: object of type S is not the same array as a member of it
}
```

There may be existing applications that use code like above for calculating
offsets of members in a structure, using pointer subtractions. This is still
undefined behavior according to the standard and code like this can be replaced
with the `offsetof` macro.

(security-putenv-stack-array)=

#### security.PutenvStackArray (C)

Finds calls to the `putenv` function which pass a pointer to a stack-allocated
(automatic) array as the argument. Function `putenv` does not copy the passed
string, only a pointer to the data is stored and this data can be read even by
other threads. Content of a stack-allocated array is likely to be overwritten
after exiting from the function.

The problem can be solved by using a static array variable or dynamically
allocated memory. Even better is to avoid using `putenv` (it has other
problems related to memory leaks) and use `setenv` instead.

The check corresponds to CERT rule
[POS34-C. Do not call putenv() with a pointer to an automatic variable as the argument](https://wiki.sei.cmu.edu/confluence/display/c/POS34-C.+Do+not+call+putenv%28%29+with+a+pointer+to+an+automatic+variable+as+the+argument).

```c
int f() {
  char env[] = "NAME=value";
  return putenv(env); // putenv function should not be called with stack-allocated string
}
```

There is one case where the checker can report a false positive. This is when
the stack-allocated array is used at `putenv` in a function or code branch that
does not return (process is terminated on all execution paths).

Another special case is if the `putenv` is called from function `main`. Here
the stack is deallocated at the end of the program and it should be no problem
to use the stack-allocated string (a multi-threaded program may require more
attention). The checker does not warn for cases when stack space of `main` is
used at the `putenv` call.

#### security.SetgidSetuidOrder (C)

When dropping user-level and group-level privileges in a program by using
`setuid` and `setgid` calls, it is important to reset the group-level
privileges (with `setgid`) first. Function `setgid` will likely fail if
the superuser privileges are already dropped.

The checker checks for sequences of `setuid(getuid())` and
`setgid(getgid())` calls (in this order). If such a sequence is found and
there is no other privilege-changing function call (`seteuid`, `setreuid`,
`setresuid` and the GID versions of these) in between, a warning is
generated. The checker finds only exactly `setuid(getuid())` calls (and the
GID versions), not for example if the result of `getuid()` is stored in a
variable.

```c
void test1() {
  // ...
  // end of section with elevated privileges
  // reset privileges (user and group) to normal user
  if (setuid(getuid()) != 0) {
    handle_error();
    return;
  }
  if (setgid(getgid()) != 0) { // warning: A 'setgid(getgid())' call following a 'setuid(getuid())' call is likely to fail
    handle_error();
    return;
  }
  // user-ID and group-ID are reset to normal user now
  // ...
}
```

In the code above the problem is that `setuid(getuid())` removes superuser
privileges before `setgid(getgid())` is called. To fix the problem the
`setgid(getgid())` should be called first. Further attention is needed to
avoid code like `setgid(getuid())` (this checker does not detect bugs like
this) and always check the return value of these calls.

This check corresponds to SEI CERT Rule [POS36-C](https://wiki.sei.cmu.edu/confluence/display/c/POS36-C.+Observe+correct+revocation+order+while+relinquishing+privileges).

(security-valist)=

#### security.VAList (C, C++)

Reports use of uninitialized (or already released) `va_list` objects and
situations where a `va_start` call is not followed by `va_end`.

```c
int test_use_after_release(int x, ...) {
  va_list va;
  va_start(va, x);
  va_end(va);
  return va_arg(va, int); // warn: va is uninitialized
}

void test_leak(int x, ...) {
  va_list va;
  va_start(va, x);
} // warn: va is leaked
```

(unix-checkers)=

### unix

POSIX/Unix checkers.

(unix-generic-options)=

#### unix generic options

These are common options that affect multiple checkers in the `unix` group.

- `unix.DynamicMemoryModeling:Optimistic`

  If set to `true`, the static analyzer assumes that all memory allocations
  and deallocations (like `malloc` or `free`) are marked with
  `ownership_holds`, `ownership_takes` and `ownership_returns`
  attributes. For more information see
  [Attributes in Clang](../AttributeReference.md#ownership-holds-ownership-returns-ownership-takes-clang-static-analyzer).
  Default value is `false`.

- `unix.DynamicMemoryModeling:ModelAllocationFailure`

  Setting this option to `true` enforces that the return value of memory
  allocation functions is tested for null by the programmer (if applicable).
  By default the analyzer does not know if a returned pointer is null or
  non-null after an allocation and access of this pointer is not reported as
  null pointer access. If the option is set to `true` the analyzer adds a
  specific execution branch where the return value is known to be null and a
  possible null pointer access can be found by other checkers. Default value of
  the option is `false`.

(unix-api)=

#### unix.API (C)

Check calls to various UNIX/Posix functions: `open, pthread_once, calloc, malloc, realloc, alloca`.

```{literalinclude} checkers/unix_api_example.c
:language: c
```

(unix-blockincriticalsection)=

#### unix.BlockInCriticalSection (C, C++)

Check for calls to blocking functions inside a critical section.
Blocking functions detected by this checker: `sleep, getc, fgets, read, recv`.
Critical section handling functions modeled by this checker:
`lock, unlock, pthread_mutex_lock, pthread_mutex_trylock, pthread_mutex_unlock, mtx_lock, mtx_timedlock, mtx_trylock, mtx_unlock, lock_guard, unique_lock, scoped_lock`.

```c
void pthread_lock_example(pthread_mutex_t *m) {
  pthread_mutex_lock(m); // note: entering critical section here
  sleep(10); // warn: Call to blocking function 'sleep' inside of critical section
  pthread_mutex_unlock(m);
}
```

```cpp
void overlapping_critical_sections(mtx_t *m1, std::mutex &m2) {
  std::lock_guard lg{m2}; // note: entering critical section here
  mtx_lock(m1); // note: entering critical section here
  sleep(10); // warn: Call to blocking function 'sleep' inside of critical section
  mtx_unlock(m1);
  sleep(10); // warn: Call to blocking function 'sleep' inside of critical section
             // still inside of the critical section of the std::lock_guard
}
```

**Limitations**

- The `trylock` and `timedlock` versions of acquiring locks are currently assumed to always succeed.
  This can lead to false positives.

```c
void trylock_example(pthread_mutex_t *m) {
  if (pthread_mutex_trylock(m) == 0) { // assume trylock always succeeds
    sleep(10); // warn: Call to blocking function 'sleep' inside of critical section
    pthread_mutex_unlock(m);
  } else {
    sleep(10); // false positive: Incorrect warning about blocking function inside critical section.
  }
}
```

(unix-chroot)=

#### unix.Chroot (C)

Check improper use of chroot described by SEI Cert C recommendation [POS05-C.
Limit access to files by creating a jail](https://wiki.sei.cmu.edu/confluence/display/c/POS05-C.+Limit+access+to+files+by+creating+a+jail).
The checker finds usage patterns where `chdir("/")` is not called immediately
after a call to `chroot(path)`.

```c
void f();

void test_bad() {
  chroot("/usr/local");
  f(); // warn: no call of chdir("/") immediately after chroot
}

 void test_bad_path() {
   chroot("/usr/local");
   chdir("/usr"); // warn: no call of chdir("/") immediately after chroot
   f();
 }

void test_good() {
  chroot("/usr/local");
  chdir("/"); // no warning
  f();
}
```

(unix-errno)=

#### unix.Errno (C)

Check for improper use of `errno`.
This checker implements partially CERT rule
[ERR30-C. Set errno to zero before calling a library function known to set errno,
and check errno only after the function returns a value indicating failure](https://wiki.sei.cmu.edu/confluence/pages/viewpage.action?pageId=87152351).
The checker can find the first read of `errno` after successful standard
function calls.

The C and POSIX standards often do not define if a standard library function
may change value of `errno` if the call does not fail.
Therefore, `errno` should only be used if it is known from the return value
of a function that the call has failed.
There are exceptions to this rule (for example `strtol`) but the affected
functions are not yet supported by the checker.
The return values for the failure cases are documented in the standard Linux man
pages of the functions and in the [POSIX standard](https://pubs.opengroup.org/onlinepubs/9699919799/).

```c
int unsafe_errno_read(int sock, void *data, int data_size) {
  if (send(sock, data, data_size, 0) != data_size) {
    // 'send' can be successful even if not all data was sent
    if (errno == 1) { // An undefined value may be read from 'errno'
      return 0;
    }
  }
  return 1;
}
```

The checker {ref}`unix-StdCLibraryFunctions` must be turned on to get the
warnings from this checker. The supported functions are the same as by
{ref}`unix-StdCLibraryFunctions`. The `ModelPOSIX` option of that
checker affects the set of checked functions.

**Parameters**

The `AllowErrnoReadOutsideConditionExpressions` option allows read of the
errno value if the value is not used in a condition (in `if` statements,
loops, conditional expressions, `switch` statements). For example `errno`
can be stored into a variable without getting a warning by the checker.

```c
int unsafe_errno_read(int sock, void *data, int data_size) {
  if (send(sock, data, data_size, 0) != data_size) {
    int err = errno;
    // warning if 'AllowErrnoReadOutsideConditionExpressions' is false
    // no warning if 'AllowErrnoReadOutsideConditionExpressions' is true
  }
  return 1;
}
```

Default value of this option is `true`. This allows save of the errno value
for possible later error handling.

**Limitations**

> - Only the very first usage of `errno` is checked after an affected function
>   call. Value of `errno` is not followed when it is stored into a variable
>   or returned from a function.
> - Documentation of function `lseek` is not clear about what happens if the
>   function returns different value than the expected file position but not -1.
>   To avoid possible false-positives `errno` is allowed to be used in this
>   case.

(unix-malloc)=

#### unix.Malloc (C)

Check for memory leaks, double free, and use-after-free problems. Traces memory managed by malloc()/free().

Custom allocation/deallocation functions can be defined using
{ref}`ownership attributes<analyzer-ownership-attrs>`.

```{literalinclude} checkers/unix_malloc_example.c
:language: c
```

(unix-mallocsizeof)=

#### unix.MallocSizeof (C)

Check for dubious `malloc` arguments involving `sizeof`.

Custom allocation/deallocation functions can be defined using
{ref}`ownership attributes<analyzer-ownership-attrs>`.

```c
void test() {
  long *p = malloc(sizeof(short));
    // warn: result is converted to 'long *', which is
    // incompatible with operand type 'short'
  free(p);
}
```

(unix-mismatcheddeallocator)=

#### unix.MismatchedDeallocator (C, C++)

Check for mismatched deallocators.

Custom allocation/deallocation functions can be defined using
{ref}`ownership attributes<analyzer-ownership-attrs>`.

```{literalinclude} checkers/mismatched_deallocator_example.cpp
:language: cpp
```

(unix-vfork)=

#### unix.Vfork (C)

Check for proper usage of `vfork`.

```c
int test(int x) {
  pid_t pid = vfork(); // warn
  if (pid != 0)
    return 0;

  switch (x) {
  case 0:
    pid = 1;
    execl("", "", 0);
    _exit(1);
    break;
  case 1:
    x = 0; // warn: this assignment is prohibited
    break;
  case 2:
    foo(); // warn: this function call is prohibited
    break;
  default:
    return 0; // warn: return is prohibited
  }

  while(1);
}
```

(unix-cstring-badsizearg)=

#### unix.cstring.BadSizeArg (C)

Check the size argument passed into C string functions for common erroneous patterns. Use `-Wno-strncat-size` compiler option to mute other `strncat`-related compiler warnings.

```c
void test() {
  char dest[3];
  strncat(dest, """""""""""""""""""""""""*", sizeof(dest));
    // warn: potential buffer overflow
}
```

(unix-cstring-notnullterminated)=

#### unix.cstring.NotNullTerminated (C)

Check for arguments which are not null-terminated strings;
applies to the `strlen`, `strcpy`, `strcat`, `strcmp` family of functions.

Only very fundamental cases are detected where the passed memory block is
absolutely different from a null-terminated string. This checker does not
find if a memory buffer is passed where the terminating zero character
is missing.

```c
void test1() {
  int l = strlen((char *)&test1); // warn
}

void test2() {
label:
  int l = strlen((char *)&&label); // warn
}
```

(unix-cstring-nullarg)=

#### unix.cstring.NullArg (C)

Check for null pointers being passed as arguments to C string functions:
`strlen, strnlen, strcpy, strncpy, strcat, strncat, strcmp, strncmp, strcasecmp, strncasecmp, wcslen, wcsnlen`.

```c
int test() {
  return strlen(0); // warn
}
```

(unix-cstring-uninitializedread)=

#### unix.cstring.UninitializedRead (C)

Check for uninitialized reads from common memory copy/manipulation functions such as:

: `memcpy, mempcpy, memmove, memcmp, strcmp, strncmp, strcpy, strlen, strsep` and many more.

```c
void test() {
 char src[10];
 char dst[5];
 memcpy(dst,src,sizeof(dst)); // warn: Bytes string function accesses uninitialized/garbage values
}
```

Limitations:

> - Due to limitations of the memory modeling in the analyzer, one can likely
>   observe some false-positives of the following kind:
>
>   > ```c
>   > void false_positive() {
>   >   int src[] = {1, 2, 3, 4};
>   >   int dst[5] = {0};
>   >   memcpy(dst, src, 4 * sizeof(int)); // false-positive:
>   >   // The 'src' buffer was correctly initialized, yet we cannot conclude
>   >   // that since the analyzer could not see a direct initialization of the
>   >   // very last byte of the source buffer.
>   > }
>   > ```
>
>   More details at the corresponding [GitHub issue](https://github.com/llvm/llvm-project/issues/43459).

(unix-stdclibraryfunctions)=

#### unix.StdCLibraryFunctions (C)

Check for calls of standard library functions that violate predefined argument
constraints. For example, according to the C standard the behavior of function
`int isalnum(int ch)` is undefined if the value of `ch` is not representable
as `unsigned char` and is not equal to `EOF`.

You can think of this checker as defining restrictions (pre- and postconditions)
on standard library functions. Preconditions are checked, and when they are
violated, a warning is emitted. Postconditions are added to the analysis, e.g.
that the return value of a function is not greater than 255. Preconditions are
added to the analysis too, in the case when the affected values are not known
before the call.

For example, if an argument to a function must be in between 0 and 255, but the
value of the argument is unknown, the analyzer will assume that it is in this
interval. Similarly, if a function mustn't be called with a null pointer and the
analyzer cannot prove that it is null, then it will assume that it is non-null.

These are the possible checks on the values passed as function arguments:
: - The argument has an allowed range (or multiple ranges) of values. The checker
    can detect if a passed value is outside of the allowed range and show the
    actual and allowed values.
  - The argument has pointer type and is not allowed to be null pointer. Many
    (but not all) standard functions can produce undefined behavior if a null
    pointer is passed, these cases can be detected by the checker.
  - The argument is a pointer to a memory block and the minimal size of this
    buffer is determined by another argument to the function, or by
    multiplication of two arguments (like at function `fread`), or is a fixed
    value (for example `asctime_r` requires at least a buffer of size 26). The
    checker can detect if the buffer size is too small and in optimal case show
    the size of the buffer and the values of the corresponding arguments.

```c
#define EOF -1
void test_alnum_concrete(int v) {
  int ret = isalnum(256); // \
  // warning: Function argument outside of allowed range
  (void)ret;
}

void buffer_size_violation(FILE *file) {
  enum { BUFFER_SIZE = 1024 };
  wchar_t wbuf[BUFFER_SIZE];

  const size_t size = sizeof(*wbuf);   // 4
  const size_t nitems = sizeof(wbuf);  // 4096

  // Below we receive a warning because the 3rd parameter should be the
  // number of elements to read, not the size in bytes. This case is a known
  // vulnerability described by the ARR38-C SEI-CERT rule.
  fread(wbuf, size, nitems, file);
}

int test_alnum_symbolic(int x) {
  int ret = isalnum(x);
  // after the call, ret is assumed to be in the range [-1, 255]

  if (ret > 255)      // impossible (infeasible branch)
    if (x == 0)
      return ret / x; // division by zero is not reported
  return ret;
}
```

Additionally to the argument and return value conditions, this checker also adds
state of the value `errno` if applicable to the analysis. Many system
functions set the `errno` value only if an error occurs (together with a
specific return value of the function), otherwise it becomes undefined. This
checker changes the analysis state to contain such information. This data is
used by other checkers, for example {ref}`unix-Errno`.

**Limitations**

The checker can not always provide notes about the values of the arguments.
Without this information it is hard to confirm if the constraint is indeed
violated. The argument values are shown if they are known constants or the value
is determined by previous (not too complicated) assumptions.

The checker can produce false positives in cases such as if the program has
invariants not known to the analyzer engine or the bug report path contains
calls to unknown functions. In these cases the analyzer fails to detect the real
range of the argument.

**Parameters**

The `ModelPOSIX` option controls if functions from the POSIX standard are
recognized by the checker.

With `ModelPOSIX=true`, many POSIX functions are modeled according to the
[POSIX standard](https://pubs.opengroup.org/onlinepubs/9699919799/). This includes ranges of parameters and possible return
values. Furthermore the behavior related to `errno` in the POSIX case is
often that `errno` is set only if a function call fails, and it becomes
undefined after a successful function call.

With `ModelPOSIX=false`, this checker follows the C99 language standard and
only models the functions that are described there. It is possible that the
same functions are modeled differently in the two cases because differences in
the standards. The C standard specifies less aspects of the functions, for
example exact `errno` behavior is often unspecified (and not modeled by the
checker).

Default value of the option is `true`.

(unix-stream)=

#### unix.Stream (C)

Check C stream handling functions:
`fopen, fdopen, freopen, tmpfile, fclose, fread, fwrite, fgetc, fgets, fputc, fputs, fprintf, fscanf, ungetc, getdelim, getline, fseek, fseeko, ftell, ftello, fflush, rewind, fgetpos, fsetpos, clearerr, feof, ferror, fileno`.

The checker maintains information about the C stream objects (`FILE *`) and
can detect error conditions related to use of streams. The following conditions
are detected:

- The `FILE *` pointer passed to the function is NULL (the single exception is
  `fflush` where NULL is allowed).
- Use of stream after close.
- Opened stream is not closed.
- Read from a stream after end-of-file. (This is not a fatal error but reported
  by the checker. Stream remains in EOF state and the read operation fails.)
- Use of stream when the file position is indeterminate after a previous failed
  operation. Some functions (like `ferror`, `clearerr`, `fseek`) are
  allowed in this state.
- Invalid 3rd ("`whence`") argument to `fseek`.

The stream operations are by this checker usually split into two cases, a success
and a failure case.
On the success case it also assumes that the current value of `stdout`,
`stderr`, or `stdin` can't be equal to the file pointer returned by `fopen`.
Operations performed on `stdout`, `stderr`, or `stdin` are not checked by
this checker in contrast to the streams opened by `fopen`.

In the case of write operations (like `fwrite`,
`fprintf` and even `fsetpos`) this behavior could produce a large amount of
unwanted reports on projects that don't have error checks around the write
operations, so by default the checker assumes that write operations always succeed.
This behavior can be controlled by the `Pedantic` flag: With
`-analyzer-config unix.Stream:Pedantic=true` the checker will model the
cases where a write operation fails and report situations where this leads to
erroneous behavior. (The default is `Pedantic=false`, where write operations
are assumed to succeed.)

```c
void test1() {
  FILE *p = fopen("foo", "r");
} // warn: opened file is never closed

void test2() {
  FILE *p = fopen("foo", "r");
  fseek(p, 1, SEEK_SET); // warn: stream pointer might be NULL
  fclose(p);
}

void test3() {
  FILE *p = fopen("foo", "r");
  if (p) {
    fseek(p, 1, 3); // warn: third arg should be SEEK_SET, SEEK_END, or SEEK_CUR
    fclose(p);
  }
}

void test4() {
  FILE *p = fopen("foo", "r");
  if (!p)
    return;

  fclose(p);
  fclose(p); // warn: stream already closed
}

void test5() {
  FILE *p = fopen("foo", "r");
  if (!p)
    return;

  fgetc(p);
  if (!ferror(p))
    fgetc(p); // warn: possible read after end-of-file

  fclose(p);
}

void test6() {
  FILE *p = fopen("foo", "r");
  if (!p)
    return;

  fgetc(p);
  if (!feof(p))
    fgetc(p); // warn: file position may be indeterminate after I/O error

  fclose(p);
}
```

**Limitations**

The checker does not track the correspondence between integer file descriptors
and `FILE *` pointers.

(osx-checkers)=

### osx

macOS checkers.

(osx-api)=

#### osx.API (C)

Check for proper uses of various Apple APIs.

```objc
void test() {
  dispatch_once_t pred = 0;
  dispatch_once(&pred, ^(){}); // warn: dispatch_once uses local
}
```

(osx-numberobjectconversion)=

#### osx.NumberObjectConversion (C, C++, ObjC)

Check for erroneous conversions of objects representing numbers into numbers.

```objc
NSNumber *photoCount = [albumDescriptor objectForKey:@"PhotoCount"];
// Warning: Comparing a pointer value of type 'NSNumber *'
// to a scalar integer value
if (photoCount > 0) {
  [self displayPhotos];
}
```

(osx-objcproperty)=

#### osx.ObjCProperty (ObjC)

Check for proper uses of Objective-C properties.

```objc
NSNumber *photoCount = [albumDescriptor objectForKey:@"PhotoCount"];
// Warning: Comparing a pointer value of type 'NSNumber *'
// to a scalar integer value
if (photoCount > 0) {
  [self displayPhotos];
}
```

(osx-seckeychainapi)=

#### osx.SecKeychainAPI (C)

Check for proper uses of Secure Keychain APIs.

```{literalinclude} checkers/seckeychainapi_example.m
:language: objc
```

(osx-cocoa-atsync)=

#### osx.cocoa.AtSync (ObjC)

Check for nil pointers used as mutexes for @synchronized.

```objc
void test(id x) {
  if (!x)
    @synchronized(x) {} // warn: nil value used as mutex
}

void test() {
  id y;
  @synchronized(y) {} // warn: uninitialized value used as mutex
}
```

(osx-cocoa-autoreleasewrite)=

#### osx.cocoa.AutoreleaseWrite

Warn about potentially crashing writes to autoreleasing objects from different autoreleasing pools in Objective-C.

(osx-cocoa-classrelease)=

#### osx.cocoa.ClassRelease (ObjC)

Check for sending 'retain', 'release', or 'autorelease' directly to a Class.

```objc
@interface MyClass : NSObject
@end

void test(void) {
  [MyClass release]; // warn
}
```

(osx-cocoa-dealloc)=

#### osx.cocoa.Dealloc (ObjC)

Warn about Objective-C classes that lack a correct implementation of -dealloc

```{literalinclude} checkers/dealloc_example.m
:language: objc
```

(osx-cocoa-incompatiblemethodtypes)=

#### osx.cocoa.IncompatibleMethodTypes (ObjC)

Warn about Objective-C method signatures with type incompatibilities.

```objc
@interface MyClass1 : NSObject
- (int)foo;
@end

@implementation MyClass1
- (int)foo { return 1; }
@end

@interface MyClass2 : MyClass1
- (float)foo;
@end

@implementation MyClass2
- (float)foo { return 1.0; } // warn
@end
```

(osx-cocoa-loops)=

#### osx.cocoa.Loops

Improved modeling of loops using Cocoa collection types.

(osx-cocoa-missingsupercall)=

#### osx.cocoa.MissingSuperCall (ObjC)

Warn about Objective-C methods that lack a necessary call to super.

```objc
@interface Test : UIViewController
@end
@implementation test
- (void)viewDidLoad {} // warn
@end
```

(osx-cocoa-nsautoreleasepool)=

#### osx.cocoa.NSAutoreleasePool (ObjC)

Warn for suboptimal uses of NSAutoreleasePool in Objective-C GC mode.

```objc
void test() {
  NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init];
  [pool release]; // warn
}
```

(osx-cocoa-nserror)=

#### osx.cocoa.NSError (ObjC)

Check usage of NSError parameters.

```objc
@interface A : NSObject
- (void)foo:(NSError """""""""""""""""""""""")error;
@end

@implementation A
- (void)foo:(NSError """""""""""""""""""""""")error {
  // warn: method accepting NSError"""""""""""""""""""""""" should have a non-void
  // return value
}
@end

@interface A : NSObject
- (BOOL)foo:(NSError """""""""""""""""""""""")error;
@end

@implementation A
- (BOOL)foo:(NSError """""""""""""""""""""""")error {
  *error = 0; // warn: potential null dereference
  return 0;
}
@end
```

(osx-cocoa-nilarg)=

#### osx.cocoa.NilArg (ObjC)

Check for prohibited nil arguments to ObjC method calls.

> - caseInsensitiveCompare:
> - compare:
> - compare:options:
> - compare:options:range:
> - compare:options:range:locale:
> - componentsSeparatedByCharactersInSet:
> - initWithFormat:

```objc
NSComparisonResult test(NSString *s) {
  NSString *aString = nil;
  return [s caseInsensitiveCompare:aString];
    // warn: argument to 'NSString' method
    // 'caseInsensitiveCompare:' cannot be nil
}
```

(osx-cocoa-nonnilreturnvalue)=

#### osx.cocoa.NonNilReturnValue

Models the APIs that are guaranteed to return a non-nil value.

(osx-cocoa-objcgenerics)=

#### osx.cocoa.ObjCGenerics (ObjC)

Check for type errors when using Objective-C generics.

```objc
NSMutableArray *names = [NSMutableArray array];
NSMutableArray *birthDates = names;

// Warning: Conversion from value of type 'NSDate *'
// to incompatible type 'NSString *'
[birthDates addObject: [NSDate date]];
```

(osx-cocoa-retaincount)=

#### osx.cocoa.RetainCount (ObjC)

Check for leaks and improper reference count management

```objc
void test() {
  NSString *s = [[NSString alloc] init]; // warn
}

CFStringRef test(char *bytes) {
  return CFStringCreateWithCStringNoCopy(
           0, bytes, NSNEXTSTEPStringEncoding, 0); // warn
}
```

(osx-cocoa-runloopautoreleaseleak)=

#### osx.cocoa.RunLoopAutoreleaseLeak

Check for leaked memory in autorelease pools that will never be drained.

(osx-cocoa-selfinit)=

#### osx.cocoa.SelfInit (ObjC)

Check that 'self' is properly initialized inside an initializer method.

```objc
@interface MyObj : NSObject {
  id x;
}
- (id)init;
@end

@implementation MyObj
- (id)init {
  [super init];
  x = 0; // warn: instance variable used while 'self' is not
         // initialized
  return 0;
}
@end

@interface MyObj : NSObject
- (id)init;
@end

@implementation MyObj
- (id)init {
  [super init];
  return self; // warn: returning uninitialized 'self'
}
@end
```

(osx-cocoa-superdealloc)=

#### osx.cocoa.SuperDealloc (ObjC)

Warn about improper use of '[super dealloc]' in Objective-C.

```objc
@interface SuperDeallocThenReleaseIvarClass : NSObject {
  NSObject *_ivar;
}
@end

@implementation SuperDeallocThenReleaseIvarClass
- (void)dealloc {
  [super dealloc];
  [_ivar release]; // warn
}
@end
```

(osx-cocoa-unusedivars)=

#### osx.cocoa.UnusedIvars (ObjC)

Warn about private ivars that are never used.

```objc
@interface MyObj : NSObject {
@private
  id x; // warn
}
@end

@implementation MyObj
@end
```

(osx-cocoa-variadicmethodtypes)=

#### osx.cocoa.VariadicMethodTypes (ObjC)

Check for passing non-Objective-C types to variadic collection
initialization methods that expect only Objective-C types.

```objc
void test() {
  [NSSet setWithObjects:@"Foo", "Bar", nil];
    // warn: argument should be an ObjC pointer type, not 'char *'
}
```

(osx-corefoundation-cferror)=

#### osx.coreFoundation.CFError (C)

Check usage of `CFErrorRef *` parameters

```c
void test(CFErrorRef *error) {
  // warn: function accepting CFErrorRef* should have a
  // non-void return
}

int foo(CFErrorRef *error) {
  *error = 0; // warn: potential null dereference
  return 0;
}
```

(osx-corefoundation-cfnumber)=

#### osx.coreFoundation.CFNumber (C)

Check for proper uses of CFNumber APIs.

```c
CFNumberRef test(unsigned char x) {
  return CFNumberCreate(0, kCFNumberSInt16Type, &x);
   // warn: 8-bit integer is used to initialize a 16-bit integer
}
```

(osx-corefoundation-cfretainrelease)=

#### osx.coreFoundation.CFRetainRelease (C)

Check for null arguments to CFRetain/CFRelease/CFMakeCollectable.

```c
void test(CFTypeRef p) {
  if (!p)
    CFRetain(p); // warn
}

void test(int x, CFTypeRef p) {
  if (p)
    return;

  CFRelease(p); // warn
}
```

(osx-corefoundation-containers-outofbounds)=

#### osx.coreFoundation.containers.OutOfBounds (C)

Checks for index out-of-bounds when using 'CFArray' API.

```c
void test() {
  CFArrayRef A = CFArrayCreate(0, 0, 0, &kCFTypeArrayCallBacks);
  CFArrayGetValueAtIndex(A, 0); // warn
}
```

(osx-corefoundation-containers-pointersizedvalues)=

#### osx.coreFoundation.containers.PointerSizedValues (C)

Warns if 'CFArray', 'CFDictionary', 'CFSet' are created with non-pointer-size values.

```c
void test() {
  int x[] = { 1 };
  CFArrayRef A = CFArrayCreate(0, (const void """""""""""""""""""""""")x, 1,
                               &kCFTypeArrayCallBacks); // warn
}
```

### Fuchsia

Fuchsia is an open source capability-based operating system currently being
developed by Google. This section describes checkers that can find various
misuses of Fuchsia APIs.

(fuchsia-handlechecker)=

#### fuchsia.HandleChecker

Handles identify resources. Similar to pointers they can be leaked,
double freed, or use after freed. This check attempts to find such problems.

```cpp
void checkLeak08(int tag) {
  zx_handle_t sa, sb;
  zx_channel_create(0, &sa, &sb);
  if (tag)
    zx_handle_close(sa);
  use(sb); // Warn: Potential leak of handle
  zx_handle_close(sb);
}
```

### WebKit

WebKit is an open-source web browser engine available for macOS, iOS and Linux.
This section describes checkers that can find issues in WebKit codebase.

Most of the checkers focus on memory management for which WebKit uses custom implementation of reference counted smartpointers.

Checkers are formulated in terms related to ref-counting:
: - *Ref-counted type* is either `Ref<T>` or `RefPtr<T>`.
  - *Ref-countable type* is any type that implements `ref()` and `deref()` methods as `RefPtr<>` is a template (i. e. relies on duck typing).
  - *Uncounted type* is ref-countable but not ref-counted type.

(webkit-refcntblbasevirtualdtor)=

#### webkit.RefCntblBaseVirtualDtor

All uncounted types used as base classes must have a virtual destructor.

Ref-counted types hold their ref-countable data by a raw pointer and allow implicit upcasting from ref-counted pointer to derived type to ref-counted pointer to base type. This might lead to an object of (dynamic) derived type being deleted via pointer to the base class type which C++ standard defines as UB in case the base class doesn't have virtual destructor `[expr.delete]`.

```cpp
struct RefCntblBase {
  void ref() {}
  void deref() {}
};

struct Derived : RefCntblBase { }; // warn
```

(webkit-nouncountedmemberchecker)=

#### webkit.NoUncountedMemberChecker

Raw pointers and references to uncounted types can't be used as class members. Only ref-counted types are allowed.

```cpp
struct RefCntbl {
  void ref() {}
  void deref() {}
};

struct Foo {
  RefCntbl * ptr; // warn
  RefCntbl & ptr; // warn
  // ...
};
```

(webkit-uncountedlambdacaptureschecker)=

#### webkit.UncountedLambdaCapturesChecker

Raw pointers and references to uncounted types can't be captured in lambdas. Only ref-counted types are allowed.

```cpp
struct RefCntbl {
  void ref() {}
  void deref() {}
};

void foo(RefCntbl* a, RefCntbl& b) {
  [&, a](){ // warn about 'a'
    do_something(b); // warn about 'b'
  };
};
```

(alpha-checkers)=

## Experimental Checkers

*These are checkers with known issues or limitations that keep them from being on by default. They are likely to have false positives. Bug reports and especially patches are welcome.*

### alpha.clone

(alpha-clone-clonechecker)=

#### alpha.clone.CloneChecker (C, C++, ObjC)

Reports similar pieces of code.

```c
void log();

int max(int a, int b) { // warn
  log();
  if (a > b)
    return a;
  return b;
}

int maxClone(int x, int y) { // similar code here
  log();
  if (x > y)
    return x;
  return y;
}
```

### alpha.core

(alpha-core-boolassignment)=

#### alpha.core.BoolAssignment (ObjC)

Warn about assigning non-{0,1} values to boolean variables.

```objc
void test() {
  BOOL b = -1; // warn
}
```

(alpha-core-c11lock)=

#### alpha.core.C11Lock

Similarly to {ref}`alpha.unix.PthreadLock <alpha-unix-PthreadLock>`, checks for
the locking/unlocking of `mtx_t` mutexes.

```cpp
mtx_t mtx1;

void bad1(void)
{
  mtx_lock(&mtx1);
  mtx_lock(&mtx1); // warn: This lock has already been acquired
}
```

(alpha-core-casttostruct)=

#### alpha.core.CastToStruct (C, C++)

Check for cast from non-struct pointer to struct pointer.

```cpp
// C
struct s {};

void test(int *p) {
  struct s *ps = (struct s *) p; // warn
}

// C++
class c {};

void test(int *p) {
  c *pc = (c *) p; // warn
}
```

(alpha-core-conversion)=

#### alpha.core.Conversion (C, C++, ObjC)

Loss of sign/precision in implicit conversions.

```c
void test(unsigned U, signed S) {
  if (S > 10) {
    if (U < S) {
    }
  }
  if (S < -10) {
    if (U < S) { // warn (loss of sign)
    }
  }
}

void test() {
  long long A = 1LL << 60;
  short X = A; // warn (loss of precision)
}
```

(alpha-core-danglingptrderef)=

#### alpha.core.DanglingPtrDeref (C++)

Check for dereferences of pointers that refer to an object whose
lifetime has already ended. Such a pointer is dangling. The checker
reports it when it is dereferenced and when it is passed to a function.
This includes a dereference in a return statement. A return statement that
does not dereference the pointer does not lead to a report. Such a case is
reported by the {ref}`core-StackAddressEscape` checker.

Each object is reported at most once on an execution path. If the same dangling
pointer is used several times then only the first use is reported.

```cpp
void test_deref() {
  int *ptr = 0;
  {
    int num = 5;
    ptr = &num;
  } // note: 'num' is destroyed here
  *ptr = 6; // warn: use of 'num' after its lifetime ended
}

int test_deref_in_return() {
  int *ptr = 0;
  {
    int num = 5;
    ptr = &num;
  } // note: 'num' is destroyed here
  return *ptr; // warn: use of 'num' after its lifetime ended
}

void test_in_scope() {
  int num = 5;
  int *ptr = &num;
  {
    *ptr = 6; // no warning, 'num' is still in scope
  }
}
```

The `-analyzer-config cfg-lifetime=true` option is a prerequisite for these
reports. Without it the checker does not report anything and no error is emitted
by the analyzer.

**Limitations**

If the analyzer cannot analyze the body of the called function, for example because
its definition is not available in the given translation unit, then a dangling
pointer passed to it is reported even if the function would never dereference
it. This can lead to false positives.

```cpp
// The definition of the function is not available that is why the analyzer
// assumes the pointer is used.
int is_null(int *p);

void argument_example() {
  int *ptr = 0;
  {
    int num = 5;
    ptr = &num;
  }
  is_null(ptr); // false positive: the pointer is compared, not dereferenced
}
```
(alpha-core-dynamictypechecker)=

#### alpha.core.DynamicTypeChecker (ObjC)

Check for cases where the dynamic and the static type of an object are unrelated.

```objc
id date = [NSDate date];

// Warning: Object has a dynamic type 'NSDate *' which is
// incompatible with static type 'NSNumber *'"
NSNumber *number = date;
[number doubleValue];
```

(alpha-core-pointerarithm)=

#### alpha.core.PointerArithm (C)

Check for pointer arithmetic on locations other than array elements.

```c
void test() {
  int x;
  int *p;
  p = &x + 1; // warn
}
```

(alpha-core-stackaddressasyncescape)=

#### alpha.core.StackAddressAsyncEscape (ObjC)

Check that addresses to stack memory do not escape the function that involves dispatch_after or dispatch_async.
This checker is a part of `core.StackAddressEscape`, but is temporarily disabled until some false positives are fixed.

```c
dispatch_block_t test_block_inside_block_async_leak() {
  int x = 123;
  void (^inner)(void) = ^void(void) {
    int y = x;
    ++y;
  };
  void (^outer)(void) = ^void(void) {
    int z = x;
    ++z;
    inner();
  };
  return outer; // warn: address of stack-allocated block is captured by a
                //       returned block
}
```

(alpha-core-stdvariant)=

#### alpha.core.StdVariant (C++)

Check if a value of active type is retrieved from an `std::variant` instance with `std::get`.
In case of bad variant type access (the accessed type differs from the active type)
a warning is emitted. Currently, this checker does not take exception handling into account.

```cpp
void test() {
  std::variant<int, char> v = 25;
  char c = stg::get<char>(v); // warn: "int" is the active alternative
}
```

(alpha-core-testafterdivzero)=

#### alpha.core.TestAfterDivZero (C)

Check for division by variable that is later compared against 0.
Either the comparison is useless or there is division by zero.

```c
void test(int x) {
  var = 77 / x;
  if (x == 0) { } // warn
}
```

(alpha-core-storetoimmutable)=

#### alpha.core.StoreToImmutable (C, C++)

Check for writes to immutable memory regions. This implements part of SEI CERT Rule ENV30-C.

This checker detects attempts to write to memory regions that are marked as immutable,
including const variables, string literals, and other const-qualified memory.

```{literalinclude} checkers/storetoimmutable_example.cpp
:language: cpp
```

**Solution**

Avoid writing to const-qualified memory regions. If you need to modify the data,
remove the const qualifier from the original declaration or use a mutable copy.

### alpha.cplusplus

(alpha-cplusplus-deletewithnonvirtualdtor)=

#### alpha.cplusplus.DeleteWithNonVirtualDtor (C++)

Reports destructions of polymorphic objects with a non-virtual destructor in their base class.

```cpp
class NonVirtual {};
class NVDerived : public NonVirtual {};

NonVirtual *create() {
  NonVirtual *x = new NVDerived(); // note: Casting from 'NVDerived' to
                                   //       'NonVirtual' here
  return x;
}

void foo() {
  NonVirtual *x = create();
  delete x; // warn: destruction of a polymorphic object with no virtual
            //       destructor
}
```

(alpha-cplusplus-invalidatediterator)=

#### alpha.cplusplus.InvalidatedIterator (C++)

Check for use of invalidated iterators.

```cpp
void bad_copy_assign_operator_list1(std::list &L1,
                                    const std::list &L2) {
  auto i0 = L1.cbegin();
  L1 = L2;
  *i0; // warn: invalidated iterator accessed
}
```

(alpha-cplusplus-iteratorrange)=

#### alpha.cplusplus.IteratorRange (C++)

Check for iterators used outside their valid ranges.

```cpp
void simple_bad_end(const std::vector &v) {
  auto i = v.end();
  *i; // warn: iterator accessed outside of its range
}
```

(alpha-cplusplus-mismatchediterator)=

#### alpha.cplusplus.MismatchedIterator (C++)

Check for use of iterators of different containers where iterators of the same container are expected.

```cpp
void bad_insert3(std::vector &v1, std::vector &v2) {
  v2.insert(v1.cbegin(), v2.cbegin(), v2.cend()); // warn: container accessed
                                                  //       using foreign
                                                  //       iterator argument
  v1.insert(v1.cbegin(), v1.cbegin(), v2.cend()); // warn: iterators of
                                                  //       different containers
                                                  //       used where the same
                                                  //       container is
                                                  //       expected
  v1.insert(v1.cbegin(), v2.cbegin(), v1.cend()); // warn: iterators of
                                                  //       different containers
                                                  //       used where the same
                                                  //       container is
                                                  //       expected
}
```

(alpha-cplusplus-smartptr)=

#### alpha.cplusplus.SmartPtr (C++)

Check for dereference of null smart pointers.

```cpp
void deref_smart_ptr() {
  std::unique_ptr<int> P;
  *P; // warn: dereference of a default constructed smart unique_ptr
}
```

### alpha.deadcode

(alpha-deadcode-unreachablecode)=

#### alpha.deadcode.UnreachableCode (C, C++)

Check unreachable code.

```cpp
// C
int test() {
  int x = 1;
  while(x);
  return x; // warn
}

// C++
void test() {
  int a = 2;

  while (a > 1)
    a--;

  if (a > 1)
    a++; // warn
}

// Objective-C
void test(id x) {
  return;
  [x retain]; // warn
}
```

### alpha.fuchsia

(alpha-fuchsia-lock)=

#### alpha.fuchsia.Lock

Similarly to {ref}`alpha.unix.PthreadLock <alpha-unix-PthreadLock>`, checks for
the locking/unlocking of fuchsia mutexes.

```cpp
spin_lock_t mtx1;

void bad1(void)
{
  spin_lock(&mtx1);
  spin_lock(&mtx1);    // warn: This lock has already been acquired
}
```

### alpha.llvm

(alpha-llvm-conventions)=

#### alpha.llvm.Conventions

Check code for LLVM codebase conventions:

- A StringRef should not be bound to a temporary std::string whose lifetime is shorter than the StringRef's.
- Clang AST nodes should not have fields that can allocate memory.

### alpha.osx

(alpha-osx-cocoa-directivarassignment)=

#### alpha.osx.cocoa.DirectIvarAssignment (ObjC)

Check for direct assignments to instance variables.

```objc
@interface MyClass : NSObject {}
@property (readonly) id A;
- (void) foo;
@end

@implementation MyClass
- (void) foo {
  _A = 0; // warn
}
@end
```

(alpha-osx-cocoa-directivarassignmentforannotatedfunctions)=

#### alpha.osx.cocoa.DirectIvarAssignmentForAnnotatedFunctions (ObjC)

Check for direct assignments to instance variables in
the methods annotated with `objc_no_direct_instance_variable_assignment`.

```objc
@interface MyClass : NSObject {}
@property (readonly) id A;
- (void) fAnnotated __attribute__((
    annotate("objc_no_direct_instance_variable_assignment")));
- (void) fNotAnnotated;
@end

@implementation MyClass
- (void) fAnnotated {
  _A = 0; // warn
}
- (void) fNotAnnotated {
  _A = 0; // no warn
}
@end
```

(alpha-osx-cocoa-instancevariableinvalidation)=

#### alpha.osx.cocoa.InstanceVariableInvalidation (ObjC)

Check that the invalidatable instance variables are
invalidated in the methods annotated with objc_instance_variable_invalidator.

```objc
@protocol Invalidation <NSObject>
- (void) invalidate
  __attribute__((annotate("objc_instance_variable_invalidator")));
@end

@interface InvalidationImpObj : NSObject <Invalidation>
@end

@interface SubclassInvalidationImpObj : InvalidationImpObj {
  InvalidationImpObj *var;
}
- (void)invalidate;
@end

@implementation SubclassInvalidationImpObj
- (void) invalidate {}
@end
// warn: var needs to be invalidated or set to nil
```

(alpha-osx-cocoa-missinginvalidationmethod)=

#### alpha.osx.cocoa.MissingInvalidationMethod (ObjC)

Check that the invalidation methods are present in classes that contain invalidatable instance variables.

```objc
@protocol Invalidation <NSObject>
- (void)invalidate
  __attribute__((annotate("objc_instance_variable_invalidator")));
@end

@interface NeedInvalidation : NSObject <Invalidation>
@end

@interface MissingInvalidationMethodDecl : NSObject {
  NeedInvalidation *Var; // warn
}
@end

@implementation MissingInvalidationMethodDecl
@end
```

(alpha-osx-cocoa-localizability-pluralmisusechecker)=

#### alpha.osx.cocoa.localizability.PluralMisuseChecker (ObjC)

Warns against using one vs. many plural pattern in code when generating localized strings.

```objc
NSString *reminderText =
  NSLocalizedString(@"None", @"Indicates no reminders");
if (reminderCount == 1) {
  // Warning: Plural cases are not supported across all languages.
  // Use a .stringsdict file instead
  reminderText =
    NSLocalizedString(@"1 Reminder", @"Indicates single reminder");
} else if (reminderCount >= 2) {
  // Warning: Plural cases are not supported across all languages.
  // Use a .stringsdict file instead
  reminderText =
    [NSString stringWithFormat:
      NSLocalizedString(@"%@ Reminders", @"Indicates multiple reminders"),
        reminderCount];
}
```

### alpha.security

(alpha-security-returnptrrange)=

#### alpha.security.ReturnPtrRange (C)

Check for an out-of-bound pointer being returned to callers.

```c
static int A[10];

int *test() {
  int *p = A + 10;
  return p; // warn
}

int test(void) {
  int x;
  return x; // warn: undefined or garbage returned
}
```

### alpha.unix

(alpha-unix-pthreadlock)=

#### alpha.unix.PthreadLock (C)

Simple lock -> unlock checker.
Applies to: `pthread_mutex_lock, pthread_rwlock_rdlock, pthread_rwlock_wrlock, lck_mtx_lock, lck_rw_lock_exclusive`
`lck_rw_lock_shared, pthread_mutex_trylock, pthread_rwlock_tryrdlock, pthread_rwlock_tryrwlock, lck_mtx_try_lock,
lck_rw_try_lock_exclusive, lck_rw_try_lock_shared, pthread_mutex_unlock, pthread_rwlock_unlock, lck_mtx_unlock, lck_rw_done`.

**Options**

- `WarnOnLockOrderReversal` (boolean). If set to true, the checker will warn
  on non-LIFO unlock order (possible lock order reversal). Defaults to false
  because detecting real lock order violations requires cross-path analysis of
  acquisition order, which the analyzer's single-path engine does not support.

```c
pthread_mutex_t mtx;

void test() {
  pthread_mutex_lock(&mtx);
  pthread_mutex_lock(&mtx);
    // warn: this lock has already been acquired
}

// The following warnings require WarnOnLockOrderReversal=true:

lck_mtx_t lck1, lck2;

void test() {
  lck_mtx_lock(&lck1);
  lck_mtx_lock(&lck2);
  lck_mtx_unlock(&lck1);
    // warn: this was not the most recently acquired lock
}

lck_mtx_t lck1, lck2;

void test() {
  if (lck_mtx_try_lock(&lck1) == 0)
    return;

  lck_mtx_lock(&lck2);
  lck_mtx_unlock(&lck1);
    // warn: this was not the most recently acquired lock
}
```

(alpha-unix-simplestream)=

#### alpha.unix.SimpleStream (C)

Check for misuses of stream APIs. Check for misuses of stream APIs: `fopen, fclose`
(demo checker, the subject of the demo ([Slides](https://llvm.org/devmtg/2012-11/Zaks-Rose-Checker24Hours.pdf) ,
[Video](https://youtu.be/kdxlsP5QVPw)) by Anna Zaks and Jordan Rose presented at the
[2012 LLVM Developers' Meeting](https://llvm.org/devmtg/2012-11/)).

```c
void test() {
  FILE *F = fopen("myfile.txt", "w");
} // warn: opened file is never closed

void test() {
  FILE *F = fopen("myfile.txt", "w");

  if (F)
    fclose(F);

  fclose(F); // warn: closing a previously closed file stream
}
```

(alpha-unix-cstring-bufferoverlap)=

#### alpha.unix.cstring.BufferOverlap (C)

Checks for overlap in two buffer arguments. Applies to: `memcpy, mempcpy, wmemcpy, wmempcpy`.

```c
void test() {
  int a[4] = {0};
  memcpy(a + 2, a + 1, 8); // warn
}
```

(alpha-unix-cstring-outofbounds)=

#### alpha.unix.cstring.OutOfBounds (C)

Check for out-of-bounds access in string functions, such as:
`memcpy, bcopy, strcpy, strncpy, strcat, strncat, memmove, memcmp, memset` and more.

This check also works with string literals, except there is a known bug in that
the analyzer cannot detect embedded NULL characters when determining the string length.

```c
void test1() {
  const char str[] = "Hello world";
  char buffer[] = "Hello world";
  memcpy(buffer, str, sizeof(str) + 1); // warn
}

void test2() {
  const char str[] = "Hello world";
  char buffer[] = "Helloworld";
  memcpy(buffer, str, sizeof(str)); // warn
}
```

### alpha.WebKit

#### alpha.webkit.ForwardDeclChecker

Check for local variables, member variables, and function arguments that are forward declared.

```cpp
struct Obj;
Obj* provide();

struct Foo {
  Obj* ptr; // warn
};

 void foo() {
   Obj* obj = provide(); // warn
   consume(obj); // warn
 }
```

(alpha-webkit-nouncheckedptrmemberchecker)=

#### alpha.webkit.MemoryUnsafeCastChecker

Check for all casts from a base type to its derived type as these might be memory-unsafe.

Example:

```cpp
class Base { };
class Derived : public Base { };

void f(Base* base) {
    Derived* derived = static_cast<Derived*>(base); // ERROR
}
```

For all cast operations (C-style casts, static_cast, reinterpret_cast, dynamic_cast), if the source type a `Base*` and the destination type is `Derived*`, where `Derived` inherits from `Base`, the static analyzer should signal an error.

This applies to:

- C structs, C++ structs and classes, and Objective-C classes and protocols.
- Pointers and references.
- Inside template instantiations and macro expansions that are visible to the compiler.

For types like this, instead of using built in casts, the programmer will use helper functions that internally perform the appropriate type check and disable static analysis.

#### alpha.webkit.NoDeleteChecker

Check that `[[clang::annotate_type("webkit.nodelete")]]` annotation does not appear on a function which could delete an object.

```cpp
void [[clang::annotate_type("webkit.nodelete")]] someFunction(RefCountable* obj) { // warn
  delete obj;
};

Foo [[clang::annotate_type("webkit.nodelete")]] trivialFunction(RefCountable* obj) {
  return obj->anotherTrivialFunction();
};
```

`[[clang::annotate_type("webkit.nodelete")]]` annotation makes the function ignored for the purpose of other WebKit smart pointer checkers.
For example, `alpha.webkit.UncountedCallArgsChecker` will ignore a function call with this annotation.

#### alpha.webkit.NoUncheckedPtrMemberChecker

Raw pointers and references to an object which supports CheckedPtr or CheckedRef can't be used as class members. Only CheckedPtr, CheckedRef, RefPtr, or Ref are allowed.

```cpp
struct CheckableObj {
  void incrementCheckedPtrCount() {}
  void decrementCheckedPtrCount() {}
};

struct Foo {
  CheckableObj* ptr; // warn
  CheckableObj& ptr; // warn
  // ...
};
```

See [WebKit Guidelines for Safer C++ Programming](https://github.com/WebKit/WebKit/wiki/Safer-CPP-Guidelines) for details.

#### alpha.webkit.NoUnretainedMemberChecker

Raw pointers and references to a NS or CF object can't be used as class members or ivars. Only RetainPtr is allowed for CF types regardless of whether ARC is enabled or disabled. Only RetainPtr or OSObjectPtr is allowed for NS types when ARC is disabled.

```cpp
struct Foo {
  NSObject *ptr; // warn
  dispatch_queue_t queue; // warn
  // ...
};
```

See [WebKit Guidelines for Safer C++ Programming](https://github.com/WebKit/WebKit/wiki/Safer-CPP-Guidelines) for details.

#### alpha.webkit.UnretainedLambdaCapturesChecker

Raw pointers and references to NS or CF types can't be captured in lambdas. Only RetainPtr is allowed for CF types regardless of whether ARC is enabled or disabled, and only RetainPtr or OSObjectPtr is allowed for NS types when ARC is disabled.

```cpp
void foo(NSObject *a, NSObject *b, dispatch_queue_t c) {
  [&, a](){ // warn about 'a'
    do_something(b); // warn about 'b'
    dispatch_queue_get_specific(c, "some"); // warn about 'c'
  };
};
```

(alpha-webkit-uncountedcallargschecker)=

#### alpha.webkit.UncountedCallArgsChecker

The goal of this rule is to make sure that lifetime of any dynamically allocated ref-countable object passed as a call argument spans past the end of the call. This applies to call to any function, method, lambda, function pointer or functor. Ref-countable types aren't supposed to be allocated on stack so we check arguments for parameters of raw pointers and references to uncounted types.

Here are some examples of situations that we warn about as they *might* be potentially unsafe. The logic is that either we're able to guarantee that an argument is safe or it's considered if not a bug then bug-prone.

> ```cpp
> RefCountable* provide_uncounted();
> void consume(RefCountable*);
>
> // In these cases we can't make sure callee won't directly or indirectly call `deref()` on the argument which could make it unsafe from such point until the end of the call.
>
> void foo1() {
>   consume(provide_uncounted()); // warn
> }
>
> void foo2() {
>   RefCountable* uncounted = provide_uncounted();
>   consume(uncounted); // warn
> }
> ```

Although we are enforcing member variables to be ref-counted by `webkit.NoUncountedMemberChecker` any method of the same class still has unrestricted access to these. Since from a caller's perspective we can't guarantee a particular member won't get modified by callee (directly or indirectly) we don't consider values obtained from members safe.

Note: It's likely this heuristic could be made more precise with fewer false positives - for example calls to free functions that don't have any parameter other than the pointer should be safe as the callee won't be able to tamper with the member unless it's a global variable.

> ```cpp
> struct Foo {
>   RefPtr<RefCountable> member;
>   void consume(RefCountable*) { /* ... */ }
>   void bugprone() {
>     consume(member.get()); // warn
>   }
> };
> ```

The implementation of this rule is a heuristic - we define a whitelist of kinds of values that are considered safe to be passed as arguments. If we can't prove an argument is safe it's considered an error.

Allowed kinds of arguments:

- values obtained from ref-counted objects (including temporaries as those survive the call too)

  ```cpp
  RefCountable* provide_uncounted();
  void consume(RefCountable*);

  void foo() {
    RefPtr<RefCountable> rc = makeRef(provide_uncounted());
    consume(rc.get()); // ok
    consume(makeRef(provide_uncounted()).get()); // ok
  }
  ```

- forwarding uncounted arguments from caller to callee

  ```cpp
  void foo(RefCountable& a) {
    bar(a); // ok
  }
  ```

  Caller of `foo()` is responsible for `a`'s lifetime.

- `this` pointer

  ```cpp
  void Foo::foo() {
    baz(this);  // ok
  }
  ```

  Caller of `foo()` is responsible for keeping the memory pointed to by `this` pointer safe.

- constants

  ```cpp
  foo(nullptr, NULL, 0); // ok
  ```

We also define a set of safe transformations which if passed a safe value as an input provide (usually it's the return value) a safe value (or an object that provides safe values). This is also a heuristic.

- constructors of ref-counted types (including factory methods)
- getters of ref-counted types
- member overloaded operators
- casts
- unary operators like `&` or `*`

#### alpha.webkit.UncheckedCallArgsChecker

The goal of this rule is to make sure that lifetime of any dynamically allocated CheckedPtr capable object passed as a call argument keeps its memory region past the end of the call. This applies to call to any function, method, lambda, function pointer or functor. CheckedPtr capable objects aren't supposed to be allocated on stack so we check arguments for parameters of raw pointers and references to unchecked types.

The rules of when to use and not to use CheckedPtr / CheckedRef are same as alpha.webkit.UncountedCallArgsChecker for ref-counted objects.

#### alpha.webkit.UncheckedLambdaCapturesChecker

Raw pointers and references to unchecked types can't be captured in lambdas. Only CheckedPtr or CheckedRef is allowed.

```cpp
struct CheckedObject {
  void incrementCheckedPtr() {}
  void decrementCheckedPtr() {}
};

void foo(CheckedObject* a, CheckedObject& b) {
  [&, a](){ // warn about 'a'
    do_something(b); // warn about 'b'
  };
};
```

#### alpha.webkit.UnretainedCallArgsChecker

The goal of this rule is to make sure that lifetime of any dynamically allocated NS or CF objects passed as a call argument keeps its memory region past the end of the call. This applies to call to any function, method, lambda, function pointer or functor. NS or CF objects aren't supposed to be allocated on stack so we check arguments for parameters of raw pointers and references to unretained types.

The rules of when to use and not to use RetainPtr or OSObjectPtr are same as alpha.webkit.UncountedCallArgsChecker for ref-counted objects.

#### alpha.webkit.UncountedLocalVarsChecker

The goal of this rule is to make sure that any uncounted local variable is backed by a ref-counted object with lifetime that is strictly larger than the scope of the uncounted local variable. To be on the safe side we require the scope of an uncounted variable to be embedded in the scope of ref-counted object that backs it.

These are examples of cases that we consider safe:

> ```cpp
> void foo1() {
>   RefPtr<RefCountable> counted;
>   // The scope of uncounted is EMBEDDED in the scope of counted.
>   {
>     RefCountable* uncounted = counted.get(); // ok
>   }
> }
>
> void foo2(RefPtr<RefCountable> counted_param) {
>   RefCountable* uncounted = counted_param.get(); // ok
> }
>
> void FooClass::foo_method() {
>   RefCountable* uncounted = this; // ok
> }
> ```

Here are some examples of situations that we warn about as they *might* be potentially unsafe. The logic is that either we're able to guarantee that a local variable is safe or it's considered unsafe.

> ```cpp
> void foo1() {
>   RefCountable* uncounted = new RefCountable; // warn
> }
>
> RefCountable* global_uncounted;
> void foo2() {
>   RefCountable* uncounted = global_uncounted; // warn
> }
>
> void foo3() {
>   RefPtr<RefCountable> counted;
>   // The scope of uncounted is not EMBEDDED in the scope of counted.
>   RefCountable* uncounted = counted.get(); // warn
> }
> ```

#### alpha.webkit.UncheckedLocalVarsChecker

The goal of this rule is to make sure that any unchecked local variable is backed by a CheckedPtr or CheckedRef with lifetime that is strictly larger than the scope of the unchecked local variable. To be on the safe side we require the scope of an unchecked variable to be embedded in the scope of CheckedPtr/CheckRef object that backs it.

These are examples of cases that we consider safe:

> ```cpp
> void foo1() {
>   CheckedPtr<RefCountable> counted;
>   // The scope of uncounted is EMBEDDED in the scope of counted.
>   {
>     RefCountable* uncounted = counted.get(); // ok
>   }
> }
>
> void foo2(CheckedPtr<RefCountable> counted_param) {
>   RefCountable* uncounted = counted_param.get(); // ok
> }
>
> void FooClass::foo_method() {
>   RefCountable* uncounted = this; // ok
> }
> ```

Here are some examples of situations that we warn about as they *might* be potentially unsafe. The logic is that either we're able to guarantee that a local variable is safe or it's considered unsafe.

> ```cpp
> void foo1() {
>   RefCountable* uncounted = new RefCountable; // warn
> }
>
> RefCountable* global_uncounted;
> void foo2() {
>   RefCountable* uncounted = global_uncounted; // warn
> }
>
> void foo3() {
>   RefPtr<RefCountable> counted;
>   // The scope of uncounted is not EMBEDDED in the scope of counted.
>   RefCountable* uncounted = counted.get(); // warn
> }
> ```

#### alpha.webkit.UnretainedLocalVarsChecker

The goal of this rule is to make sure that any NS or CF local variable is backed by a RetainPtr or OSObjectPtr with lifetime that is strictly larger than the scope of the unretained local variable. To be on the safe side we require the scope of an unretained variable to be embedded in the scope of RetainPtr or OSObjectPtr object that backs it.

The rules of when to use and not to use RetainPtr or OSObjectPtr are same as alpha.webkit.UncountedCallArgsChecker for ref-counted objects.

These are examples of cases that we consider safe:

> ```cpp
> void foo1() {
>   RetainPtr<NSObject> retained;
>   // The scope of unretained is EMBEDDED in the scope of retained.
>   {
>     NSObject* unretained = retained.get(); // ok
>   }
> }
>
> void foo2(RetainPtr<NSObject> retained_param) {
>   NSObject* unretained = retained_param.get(); // ok
> }
>
> void FooClass::foo_method() {
>   NSObject* unretained = this; // ok
> }
> ```

Here are some examples of situations that we warn about as they *might* be potentially unsafe. The logic is that either we're able to guarantee that a local variable is safe or it's considered unsafe.

> ```cpp
> void foo1() {
>   NSObject* unretained = [[NSObject alloc] init]; // warn
> }
>
> NSObject* global_unretained;
> void foo2() {
>   NSObject* unretained = global_unretained; // warn
> }
>
> void foo3() {
>   RetainPtr<NSObject> retained;
>   // The scope of unretained is not EMBEDDED in the scope of retained.
>   NSObject* unretained = retained.get(); // warn
> }
> ```

#### webkit.RetainPtrCtorAdoptChecker

The goal of this rule is to make sure the constructors of RetainPtr and OSObjectPtr as well as adoptNS, adoptCF, and adoptOSObject are used correctly.
When creating a RetainPtr or OSObjectPtr with +1 semantics, adoptNS, adoptCF, or adoptOSObject should be used, and in +0 semantics, RetainPtr or OSObjectPtr constructor should be used.
Warn otherwise.

These are examples of cases that we consider correct:

> ```cpp
> RetainPtr ptr = adoptNS([[NSObject alloc] init]); // ok
> RetainPtr ptr = CGImageGetColorSpace(image); // ok
> OSObjectPtr ptr = adoptOSObject(dispatch_queue_create("some queue", nullptr)); // ok
> ```

Here are some examples of cases that we consider incorrect use of RetainPtr constructor and adoptCF

> ```cpp
> RetainPtr ptr = [[NSObject alloc] init]; // warn
> auto ptr = adoptCF(CGImageGetColorSpace(image)); // warn
> OSObjectPtr ptr = dispatch_queue_create("some queue", nullptr); // warn
> ```

## Debug Checkers

(debug-checkers)=

### debug

Checkers used for debugging the analyzer.
{doc}`developer-docs/DebugChecks` page contains a detailed description.

(debug-analysisorder)=

#### debug.AnalysisOrder

Print callbacks that are called during analysis in order.

(debug-configdumper)=

#### debug.ConfigDumper

Dump config table.

(debug-dumpcfg-display)=

#### debug.DumpCFG Display

Control-Flow Graphs.

(debug-dumpcallgraph)=

#### debug.DumpCallGraph

Display Call Graph.

(debug-dumpcalls)=

#### debug.DumpCalls

Print calls as they are traversed by the engine.

(debug-dumpdominators)=

#### debug.DumpDominators

Print the dominance tree for a given CFG.

(debug-dumplivevars)=

#### debug.DumpLiveVars

Print results of live variable analysis.

(debug-dumptraversal)=

#### debug.DumpTraversal

Print branch conditions as they are traversed by the engine.

(debug-exprinspection)=

#### debug.ExprInspection

Check the analyzer's understanding of expressions.

(debug-stats)=

#### debug.Stats

Emit warnings with analyzer statistics.

(debug-tainttest)=

#### debug.TaintTest

Mark tainted symbols as such.

(debug-viewcfg)=

#### debug.ViewCFG

View Control-Flow Graphs using GraphViz.

(debug-viewcallgraph)=

#### debug.ViewCallGraph

View Call Graph using GraphViz.

(debug-viewexplodedgraph)=

#### debug.ViewExplodedGraph

View Exploded Graphs using GraphViz.
