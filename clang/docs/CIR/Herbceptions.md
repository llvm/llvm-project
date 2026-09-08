# Herbceptions (throws/return_failure) in ClangIR

Herbceptions are an error handling mechanism for C++ that uses a `{T, i1}` return
type to carry both the success value and a failure discriminant. This document
describes how herbceptions are implemented in ClangIR.

## Overview

A function declared with `throws` or `return_failure{E}` has its return type transformed
into a `{T, i1}` record, where `T` is the original return type (or the error
type for `void` functions) and `i1` is the failure discriminant.

```cpp
int foo(int x) throws {
  if (x < 0) throw throws std::my_errc{x};
  return x + 1;
}
```

The function above is lowered to a CIR function returning `{{ptr, i64}, i1}`
(because `int` is smaller than the implicit `std::error` type, the payload is
promoted to the error type).

## CIR Representation

### Function Signature

Herbception functions have the `cir.throws` attribute and return a `{T, i1}`
record:

```mlir
cir.func no_inline @_Z3fooi(%arg0: !s32i) -> !rec_anon_struct1
  attributes {cir.throws, ...} {
  ...
}
```

### Key Operations

- `cir.throws` - Attribute marking a function as a herbception function
- `cir.goto` - Used to branch to error handlers
- `cir.label` - Marks error handler blocks
- `cir.if` - Routes the discriminant to either the success or error path

## Error Handling Flow

### Throwing Errors

A `throw throws` expression stores the error value into the return slot and
returns with the discriminant set to true:

```mlir
// Store error payload
cir.store %error, %retval : !rec_error, !cir.ptr<!rec.error>
// Set discriminant to true
cir.store #true, %disc : !cir.bool, !cir.ptr<!cir.bool>
// Return with {error, true}
cir.return %shaped_result : !rec_anon_struct1
```

### Catching Errors

A `try/catch throws` block routes errors to the handler:

```mlir
%result = cir.call @foo() {cir.throws} : () -> !rec_anon_struct1
%disc = cir.get_member %result[1] {name = ""} : ... -> !cir.ptr<!cir.bool>
%disc_val = cir.load %disc : !cir.ptr<!cir.bool>, !cir.bool
cir.if %disc_val {
  // Error path - route to handler
  cir.goto "catch.throws.0"
}
^bb:
  // Success path
  cir.goto "herb.try.cont.0"
}
```

### Auto-propagation with try(expr)

The `try(expr)` expression auto-propagates errors from a throws call:

```cpp
int caller(int x) throws {
  return try(foo(x));
}
```

This is lowered to extract the discriminant and return early on error:

```mlir
%result = cir.call @foo() {cir.throws} : () -> !rec_anon_struct1
%disc = cir.get_member %result[1] {name = ""} : ... -> !cir.ptr<!cir.bool>
%disc_val = cir.load %disc : !cir.ptr<!cir.bool>, !cir.bool
cir.if %disc_val {
  // Auto-propagate: wrap with disc=true and return
  %payload = cir.get_member %result[0] {name = ""} : ... -> !cir.ptr<!s32i>
  %payload_val = cir.load %payload : !cir.ptr<!s32i>, !s32i
  %wrapped = ... // wrapHerbceptionReturnValue(loc, payload_val, true)
  cir.return %wrapped : !rec_anon_struct1
}
```

### catch return_failure(expr)

The `catch return_failure(expr)` expression builds an `either{T, E}` aggregate:

```cpp
auto e = catch return_failure(bar(x));
return !e.failed ? e.value * 2 : e.error;
```

This is lowered to extract both the payload and discriminant and build the
aggregate struct.

## Special Functions

### Constructors

Constructors can have `throws` specifications. They use the same `{T, i1}`
return type mechanism:

```cpp
struct Foo {
  Foo(int x) throws : x(x) {}
};
```

### Copy/Move Constructors

Copy and move constructors can also have `throws`:

```cpp
struct Foo {
  Foo(const Foo& other) throws : x(other.x) {}
  Foo(Foo&& other) throws : x(other.x) {}
};
```

### Operator Overloading

Operators can have `throws` specifications:

```cpp
struct Bar {
  Bar operator+(const Bar& other) throws { ... }
  int& operator[](int idx) throws { ... }
};
```

### Destructors

Destructors **cannot** have `throws` specifications. This is enforced by
Sema:

```cpp
struct Bad {
  ~Bad() throws {} // Error: destructor cannot have a throws specifier
};
```

## Interop with Classic C++ Exceptions

Herbceptions can coexist with classic C++ exceptions. A `catch throws`
handler with a legacy conversion expression catches both herbception errors
and classic C++ exceptions:

```cpp
void callee() noexcept {
  try {
    bar();  // throws function
  } catch throws(std::error e) {
    // Catches both herbception errors and C++ exceptions
  }
}
```

## Implementation Files

- `clang/lib/CIR/CodeGen/CIRGenException.cpp` - Exception handling lowering
- `clang/lib/CIR/CodeGen/CIRGenCall.cpp` - Call lowering with herbception routing
- `clang/lib/CIR/CodeGen/CIRGenFunction.cpp` - Function-level herbception support
- `clang/lib/CIR/CodeGen/CIRGenTypes.cpp` - Type conversion for herbception signatures
- `clang/lib/CIR/CodeGen/CIRGenStmt.cpp` - Statement-level lowering
- `clang/lib/CIR/CodeGen/CIRGenExprScalar.cpp` - Scalar expression visitors
- `clang/lib/CIR/CodeGen/CIRGenExprAggregate.cpp` - Aggregate expression visitors

## Tests

Herbception tests are located in:
- `clang/test/CodeGen/herbception-*.cpp` - Classic codegen tests
- `clang/test/CIR/CodeGen/herbception-*.cpp` - ClangIR tests
