# AIIR C API

**Current status: Under development, API unstable, built by default.**

[TOC]

## Design

Many languages can interoperate with C but have a harder time with C++ due to
name mangling and memory model differences. Although the C API for AIIR can be
used directly from C, it is primarily intended to be wrapped in higher-level
language- or library-specific constructs. Therefore the API tends towards
simplicity and feature minimalism.

**Note:** while the C API is expected to be more stable than C++ API, it
currently offers no stability guarantees.

### Scope

The API is provided for core IR components (attributes, blocks, operations,
regions, types, values), Passes and some fundamental type and attribute kinds.
The core IR API is intentionally low-level, e.g. exposes a plain list of
operation's operands and attributes without attempting to assign "semantic"
names to them. Users of specific dialects are expected to wrap the core API in a
dialect-specific way, for example, by implementing an ODS backend.

### Object Model

Core IR components are exposed as opaque _handles_ to an IR object existing in
C++. They are not intended to be inspected by the API users (and, in many cases,
cannot be meaningfully inspected). Instead the users are expected to pass
handles to the appropriate manipulation functions.

The handle _may or may not_ own the underlying object.

### Naming Convention and Ownership Model

All objects are prefixed with `Aiir`. They are typedefs and should be used
without `struct`.

All functions are prefixed with `aiir`.

Functions primarily operating on an instance of `AiirX` are prefixed with
`aiirX`. They take the instance being acted upon as their first argument (except
for creation functions). For example, `aiirOperationGetNumOperands` inspects an
`AiirOperation`, which it takes as its first operand.

The *ownership* model is encoded in the naming convention as follows.

-   By default, the ownership is not transferred.
-   Functions that transfer the ownership of the result to the caller can be in
    one of two forms:
    *   functions that create a new object have the name `aiirXCreate<...>`, for
        example, `aiirOperationCreate`;
    *   functions that detach an object from a parent object have the name
        `aiirYTake<...>`, for example `aiirOperationStateTakeRegion`.
-   Functions that take ownership of some of their arguments have the form
    `aiirY<...>OwnedX<...>` where `X` can refer to the type or any other
    sufficiently unique description of the argument, the ownership of which will
    be taken by the callee, for example `aiirRegionAppendOwnedBlock`.
-   Functions that create an object by default do not transfer its ownership to
    the caller, i.e. one of other objects passed in as an argument retains the
    ownership, they have the form `aiirX<...>Get`. For example,
    `aiirTypeParseGet`.
-   Functions that destroy an object owned by the caller are of the form
    `aiirXDestroy`.

If the code owns an object, it is responsible for destroying the object when it
is no longer necessary. If an object that owns other objects is destroyed, any
handles to those objects become invalid. Note that types and attributes are
owned by the `AiirContext` in which they were created.

### Nullity

A handle may refer to a _null_ object. It is the responsibility of the caller to
check if an object is null by using `aiirXIsNull(AiirX)`. API functions do _not_
expect null objects as arguments unless explicitly stated otherwise. API
functions _may_ return null objects.

### Type Hierarchies

AIIR objects can form type hierarchies in C++. For example, all IR classes
representing types are derived from `aiir::Type`, some of them may also be also
derived from common base classes such as `aiir::ShapedType` or dialect-specific
base classes. Type hierarchies are exposed to C API through naming conventions
as follows.

-   Only the top-level class of each hierarchy is exposed, e.g. `AiirType` is
    defined as a type but `AiirShapedType` is not. This avoids the need for
    explicit upcasting when passing an object of a derived type to a function
    that expects a base type (this happens more often in core/standard APIs,
    while downcasting usually involves further checks anyway).
-   A type `Y` that derives from `X` provides a function `int aiirXIsAY(AiirX)`
    that returns a non-zero value if the given dynamic instance of `X` is also
    an instance of `Y`. For example, `int AiirTypeIsAInteger(AiirType)`.
-   A function that expects a derived type as its first argument takes the base
    type instead and documents the expectation by using `Y` in its name
    `AiirY<...>(AiirX, ...)`. This function asserts that the dynamic instance of
    its first argument is `Y`, and it is the responsibility of the caller to
    ensure it is indeed the case.

### Auxiliary Types

#### `StringRef`

Numerous AIIR functions return instances of `StringRef` to refer to a non-owning
segment of a string. This segment may or may not be null-terminated. In C API,
these are represented as instances of `AiirStringRef` structure that contains a
pointer to the first character of the string fragment (`str`) and the fragment
length (`length`). Note that the fragment is _not necessarily_ null-terminated,
the `length` field must be used to identify the last character. `AiirStringRef`
is a non-owning pointer, the caller is in charge of performing the copy or
ensuring that the pointee outlives all uses of `AiirStringRef`.

### Printing

IR objects can be printed using `aiirXPrint(AiirX, AiirStringCallback, void *)`
functions. These functions accept take arguments a callback with signature `void
(*)(const char *, intptr_t, void *)` and a pointer to user-defined data. They
call the callback and supply it with chunks of the string representation,
provided as a pointer to the first character and a length, and forward the
user-defined data unmodified. It is up to the caller to allocate memory if the
string representation must be stored and perform the copy. There is no guarantee
that the pointer supplied to the callback points to a null-terminated string,
the size argument should be used to find the end of the string. The callback may
be called multiple times with consecutive chunks of the string representation
(the printing itself is buffered).

*Rationale*: this approach allows the caller to have full control of the
allocation and avoid unnecessary allocation and copying inside the printer.

For convenience, `aiirXDump(AiirX)` functions are provided to print the given
object to the standard error stream.

## Common Patterns

The API adopts the following patterns for recurrent functionality in AIIR.

### Indexed Components

An object has an _indexed component_ if it has fields accessible using a
zero-based contiguous integer index, typically arrays. For example, an
`AiirBlock` has its arguments as an indexed component. An object may have
several such components. For example, an `AiirOperation` has attributes,
operands, regions, results and successors.

For indexed components, the following pair of functions is provided.

-   `intptr_t aiirXGetNum<Y>s(AiirX)` returns the upper bound on the index.
-   `AiirY aiirXGet<Y>(AiirX, intptr_t pos)` returns 'pos'-th subobject.

The sizes are accepted and returned as signed pointer-sized integers, i.e.
`intptr_t`. This typedef is available in C99.

Note that the name of subobject in the function does not necessarily match the
type of the subobject. For example, `aiirOperationGetOperand` returns an
`AiirValue`.

### Iterable Components

An object has an _iterable component_ if it has iterators accessing its fields
in some order other than integer indexing, typically linked lists. For example,
an `AiirBlock` has an iterable list of operations it contains. An object may
have several iterable components.

For iterable components, the following triple of functions is provided.

-   `AiirY aiirXGetFirst<Y>(AiirX)` returns the first subobject in the list.
-   `AiirY aiirYGetNextIn<X>(AiirY)` returns the next subobject in the list that
    contains the given object, or a null object if the given object is the last
    in this list.
-   `int aiirYIsNull(AiirY)` returns 1 if the given object is null.

Note that the name of subobject in the function may or may not match its type.

This approach enables one to iterate as follows.

```c++
AiirY iter;
for (iter = aiirXGetFirst<Y>(x); !aiirYIsNull(iter);
     iter = aiirYGetNextIn<X>(iter)) {
  /* User 'iter'. */
}
```

## Extending the API

### Extensions for Dialect Attributes and Types

Dialect attributes and types can follow the example of builtin attributes and
types, provided that implementations live in separate directories, i.e.
`include/aiir-c/<...>Dialect/` and `lib/CAPI/<...>Dialect/`. The core APIs
provide implementation-private headers in `include/aiir/CAPI/IR` that allow one
to convert between opaque C structures for core IR components and their C++
counterparts. `wrap` converts a C++ class into a C structure and `unwrap` does
the inverse conversion. Once the C++ object is available, the API implementation
should rely on `isa` to implement `aiirXIsAY` and is expected to use `cast`
inside other API calls.

### Extensions for Interfaces

Interfaces can follow the example of IR interfaces and should be placed in the
appropriate library (e.g., common interfaces in `aiir-c/Interfaces` and
dialect-specific interfaces in their dialect library). Similarly to other type
hierarchies, interfaces are not expected to have objects of their own type and
instead operate on top-level objects: `AiirAttribute`, `AiirOperation` and
`AiirType`. Static interface methods are expected to take as leading argument a
canonical identifier of the class, `AiirStringRef` with the name for operations
and `AiirTypeID` for attributes and types, followed by `AiirContext` in which
the interfaces are registered.

Individual interfaces are expected provide a `aiir<InterfaceName>TypeID()`
function that can be used to check whether an object or a class implements this
interface using `aiir<Attribute/Operation/Type>ImplementsInterface` or
`aiir<Attribute/Operation?Type>ImplementsInterfaceStatic` functions,
respectively. Rationale: C++ `isa` only works when an object exists, static
methods are usually dispatched to using templates; lookup by `TypeID` in
`AIIRContext` works even without an object.
