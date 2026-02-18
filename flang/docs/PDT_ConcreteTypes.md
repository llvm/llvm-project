<!--===- docs/PDT_ConcreteTypes.md

   Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
   See https://llvm.org/LICENSE.txt for license information.
   SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

-->

# Parameterized Derived Types with LEN Type Parameters

## The LEN Type Parameter Runtime Problem

For PDTs with LEN parameters, component offsets can depend either entirely or
partially on runtime values:

```fortran
! Simple PDT with LEN type parameter
type :: pdt(N)
  integer, len   :: N
  character(N*2) :: str        ! size = 2*N bytes
  integer        :: after_str  ! offset within PDT depends on runtime val of N
end type

! More complex PDT with both LEN and KIND type parameters
type :: pdt2(M,N,K)
  integer, len  :: M,N
  integer, kind :: K
  real          :: R1(M)  ! Uses runtime value of M
  real(kind=K)  :: R2     ! Offset depends on runtime M * size of default REAL
  integer       :: VAL(N) ! Offset depends on previous members, and must account
                          ! for the compile-time KIND value as well. 
end type

! Nested, with LEN type parameter
type :: nestedType(A, B)
  integer, len  :: A,B
  type(pdt(A)) :: X ! Nested, pass "A" to component
  type(pdt(B)) :: Y ! Nested, pass "B" to component
end type
```



## Proposed Solution: Concrete Type Instantiation

The concrete type approach creates resolved type descriptions lazily at runtime
and caches them. All instances with identical LEN values share the same
concrete type, providing O(1) component access after initial instantiation.

For any specific set of LEN values, the type layout is fully determined.
For example, all instances of `pdt(10)` will have identical component offsets.
Rather than recomputing offsets on each access, we resolve them once and cache
the result.


### Architecture Overview

```
                    +---------------------------+
                    | DerivedType "pdt"         |
                    | (generic/uninstantiated)  |
                    | offset_: 0 (placeholder)  |<------------------+
                    | numLenParams: 1           |                   |
                    | sizeInBytes_: 0           |                   |
                    | uninstantiated_: nullptr  |<----------------+ |
                    +---------------------------+                 | |
            .                                                     | |
            .                                                     | |
+------------------------+     +-------------------------------+  | |
| Descriptor             |     | Concrete DerivedType          |  | |
| type(pdt(N=10)) :: A   |     | Hash: 0xABC123                |  | |
| len_ = {10}            |     | 0x7fff8a003000 (heap)         |  | |
| derivedType_:----------|---->| offset_: {0, 4, 24}           |  | |
+------------------------+     | sizeInBytes_: 28              |  | |
            .                  | uninstantiated_: -------------|--+ |
            .                  | numLenParams: 1               |    |
            .                  +-------------------------------+    |
+------------------------+               ^                          |
| Descriptor             |               |                          |
| type(pdt(N=10)) :: B   |               |                          |
| len_[] = {10}          |               |                          |
| derivedType_: ---------|---------------+                          |
+------------------------+                                          |
            .                                                       |
+------------------------+     +----------------------------------+ |
| Descriptor             |     | Concrete DerivedType             | |
| type(pdt(N=20)) :: C   |     | 0x7fff8a004000 (heap)            | |
| len_ = {20}            |     | offset_: {0, 4,  44}             | |
| derivedType_:----------|---->| sizeInBytes_: 48                 | |
+------------------------+     | uninstantiated_: ----------------|-+
            .                  | numLenParams: 1                  |
            .                  +----------------------------------+
```

## KIND Type Parameter Instantiation

Flang already uses an instantiation pattern for KIND type parameters that
serves as our initial model for LEN parameter handling.

For KIND parameters, the compiler generates a separate `DerivedType` at compile
time for each unique KIND instantiation. Each instantiated type points back to
the original uninstantiated type via `uninstantiated_`:

```
+--------------------+                  +---------------------------+
| Descriptor         |                  | DerivedType "pdt"         |
| type(pdt(4)) :: A  |                  | (generic/uninstantiated)  |
| derivedType_: -----|-+                | uninstantiated_: nullptr  |
+--------------------+ |                +---------------------------+
                       |                             ^           ^
+-------------------+  |                             |           |
| Descriptor        |  |   +----------------------+  |           |
| type(pdt(4)) :: B |  +-> | DerivedType "pdt.k4" |  |           |
| derivedType_: ----|----> | 0x7fff8a001000 (heap)|  |           |
+-------------------+      | uninstantiated_:-----|--+           |
                           +----------------------+              |
                                                                 |
+-------------------+                  +----------------------+  |
| Descriptor        |                  | DerivedType "pdt.k8" |  |
| type(pdt(8)) :: C |                  | 0x7fff8a002000 (heap)|  |
| derivedType_: ----|----------------->| uninstantiated_:-----|--+
+-------------------+                  +----------------------+
```

Here, descriptors `A` and `B` both store the same pointer (`0x7fff8a001000`) in
their `derivedType_` field, referencing a single shared generic `DerivedType`
structure.

The `SameTypeAs` intrinsic in `derived-api` uses the pattern: if direct pointer
comparison fails, it compares `uninstantiatedType()` pointers which allows for
Type matching across module boundaries.

For LEN parameters, we apply the same pattern but create concrete types at
runtime (see Runtime Instantiation Algorithm):

- Concrete type's `uninstantiated_` points to the generic type
- Cache key: `(genericType, len_values...)` for uniqueing the DerivedType
- Use the existing `SAME_TYPE_AS` logic


## Data Structures

### Value Class

The existing `Value` class already supports LEN-dependent expressions:

```cpp
class Value {
public:
  enum class Genre : std::uint8_t {
    Deferred = 1,      // runtime-determined (ALLOCATABLE, POINTER)
    Explicit = 2,      // constant value
    LenParameter = 3,  // references LEN parameter by index
  };
  ...
  RT_API_ATTRS common::optional<TypeParameterValue>
    GetValue(const Descriptor *) const;
  ...
private:
  ...
  Genre genre_{Genre::Explicit};
  TypeParameterValue value_{0};
};
```

When `genre_ == LenParameter`, `value_` is an index into the descriptor's
`len_` Flexible array. The `GetValue()` method resolves this at runtime.

### Component Class

Each component of a derived type is described by a `Component`:

```cpp
class Component {
  ...
  StaticDescriptor<0> name_;
  Genre genre_;                    // Data, Pointer, Allocatable, Automatic
  std::uint8_t category_;          // TypeCategory
  std::uint8_t kind_;
  std::uint8_t rank_;
  MemorySpace memorySpace_;
  std::uint8_t alignment_;         // new field; log2 of targ-specific alignment
  std::uint8_t padding_[2];
  std::uint64_t offset_;           // byte offset within instance
  Value characterLen_;             // for CHARACTER components
  StaticDescriptor<0> derivedType_;
  StaticDescriptor<1> lenValue_;   // LEN values for nested PDTs
  StaticDescriptor<2> bounds_;     // array bounds (can be LEN-dependent)
  const char *initialization_;
};
```

The `characterLen_` field and the `Value` elements within `bounds_` use the
`Value::Genre::LenParameter` mechanism to express LEN-dependent sizes and
extents.

The `alignment_` field stores `log2(alignment)` for correct layout
computation on heterogeneous systems (CPU/GPU).

The `Component` class exposes a `SetOffset()` method so that
`ResolveComponentOffsets` can patch each component's byte offset in the
concrete type copy.

### DescriptorAddendum

Per-instance LEN values are stored in the descriptor's addendum:

```cpp
class DescriptorAddendum {
  ...
  const typeInfo::DerivedType *derivedType_;
  FlexibleArray<typeInfo::TypeParameterValue> len_; // must be last component
};
```

The `len_` field is changed from a fixed-size `TypeParameterValue len_[1]`
array to `FlexibleArray<TypeParameterValue>`. This template wraps a single
inline element `len_ntry_` and provides `operator[]` via pointer arithmetic in
order to emulate a C flexible array member.

Note that this is a slightly different `FlexibleArray` from the one in
`ISO_Fortran_binding.h` used for `dim`; that version inherits from `T` and
indexes via `this`-pointer arithmetic.

## Lowering Strategy

All LEN parameter expressions are evaluated at runtime and stored in the
descriptor's `len_` Flexible array. Thus, a definition such as:

```fortran
type(pdt(4, n+1, m*2)) :: x
```

will lower to code similar to:

```
%len0 = arith.constant 4
%len1 = arith.addi %n, %c1
%len2 = arith.muli %m, %c2

call @AllocatableSetDerivedLength(%desc, 0, %len0)
call @AllocatableSetDerivedLength(%desc, 1, %len1)
call @AllocatableSetDerivedLength(%desc, 2, %len2)
call @AllocatableAllocate(%desc)  // internally calls GetConcreteType
```


## Runtime Instantiation Algorithm

### GetConcreteType Entry Point

The runtime provides `GetConcreteType(genericType, instance, terminator)`:

```cpp
const DerivedType *
GetConcreteType(const DerivedType &genericType,
                const Descriptor  &instance,
                Terminator &terminator);
```

1. If `genericType.LenParameters() == 0`, return `&genericType`
2. If `genericType.uninstantiatedType()` is neither `&genericType` nor `nullptr`,
 it is already a Concrete type; return `&genericType`
3. Compute hash from `(genericType*, len_values...)`
4. Check cache; if found, return cached concrete type
5. Create new concrete type with resolved offsets
6. Insert into cache; return pointer

Step 2 handles the case where a descriptor's `derivedType_` was previously set
to a concrete type, and a subsequent operation (e.g., `AllocatableAllocate`)
passes that already-resolved type back to `GetConcreteType`. Concrete types
have `uninstantiatedType_` pointing to the original generic; generic types
point to nothing (nullptr).


### Concrete Type Creation

When creating a new concrete type:

1. Allocate `sizeof(DerivedType) + numComponents * sizeof(Component)`
2. `memcpy` the generic type (**Assumption: generic uses immutable metadata**)
3. Duplicate the components of the `Component[]` array
4. Patch the pointer members of `DerivedType`:
   - Set `component_` descriptor to point to new Component array
   - Set `uninstantiated_` to point back to generic type
5. For each component, resolve/compute actual offset based on alignment and
   sizes
6. Set sizeInBytes to the final padded size of the resolved type


### Offset Resolution

Component offsets are computed sequentially, respecting alignment. Each
component's size is determined by its genre:

- **Non-Data genres** (Allocatable, Pointer, Automatic): the component stores a
  Descriptor, so its size is `Descriptor::SizeInBytes(rank, ...)` and alignment
  is `alignof(Descriptor)`.
- **Nested PDT with LEN parameters** (Data genre, Derived category):
  `GetConcreteType` is called recursively, depth-first, to resolve the inner
  type. The resolved `sizeInBytes` is then used as the element size, and the
  component's `derivedType_` pointer is updated to the concrete type.
- **All other Data components**: element byte size times element count.

For array components, the element count is computed from the bounds - which may
themselves be nested LEN-dependent `Value` expressions.

```cpp
// Simplified view of ResolveComponentOffsets
std::size_t currentOffset = 0;
for (Component &comp : components) {
    std::size_t alignment = comp.alignment();
    currentOffset = alignTo(currentOffset, alignment);
    comp.SetOffset(currentOffset);
    currentOffset += componentSize;  // determined by genre (see above)
}
sizeInBytes = alignTo(currentOffset, maxAlignment);
```

### Recursion and Self-Referential PDTs

A PDT may reference itself through Pointer or Allocatable components:

```fortran
type :: trecurse(X)
  integer, len :: X
  type(trecurse(X+1)), pointer :: P
end type
```
The Fortran standard (F2023, C749) requires self-referential components to be
`POINTER` or `ALLOCATABLE`, thus `GetConcreteType` never encounters cyclic
recursion. Pointer/Allocatable components store a fixed-size `Descriptor`
whose size depends only on rank and DescriptorAddendum - not on the target
type's layout. Data-genre components are stored inline, so nested PDTs do
trigger a recursive `GetConcreteType` call, but this always terminates because
inline components cannot be self-referential.


```cpp
    if (comp.genre() != Component::Genre::Data) {
      // Non-Data genres (Allocatable, Pointer, Automatic): store a Descriptor
      const DerivedType *derivedComp = comp.derivedType();
      componentSize = Descriptor::SizeInBytes(
          comp.rank(), true, derivedComp ? derivedComp->LenParameters() : 0);
      alignment = alignof(Descriptor);
    } else if (...)
```

For the `trecurse` example, resolving `X=5` processes component `P` as a
Pointer (fixed ~48-byte `Descriptor`), with no recursion into `trecurse(X+1)`.
The concrete type for `trecurse(6)` is created lazily only if `P` is later
allocated. Pointer association such as `foo%P => target` is a shallow
descriptor copy and does not trigger concrete type creation.

### LEN Expression Encoding

The component's `lenValue_` array maps parent LEN params to child LEN params.
A bare reference like `type(trecurse(X))` is straightforward:
`Value(genre=LenParameter, value=0)` indexing the parent's parameter `0`.
However, `X+1` is an expression, and the Value class can only encode a
single parameter index or a constant - not computed expressions.

For Data-genre components with such expressions, the compiler will need to
emit a small compiler-generated function which the runtime can call during
`ResolveComponentOffsets`. Pointer/Allocatable components avoid this because
the compiler evaluates LEN expressions inline at the allocation or assignment
site; their `lenValue_` arrays are not consulted during offset resolution.


## Cache Design

The cache uses a custom dynamically-sized hash table (`ConcreteTypeCache`) that
relies only on C-style memory management (`malloc`, `calloc`, `free`), and uses
the hash value directly as the lookup key:

```cpp
// Hashing, based on Boost's hash_combine.
std::uint64_t
ComputeConcreteTypeHash(const DerivedType &genericType,
                        const DescriptorAddendum &addendum,
                        std::size_t numLenParams)
{
  std::uint64_t hash = reinterpret_cast<std::uintptr_t>(&genericType);
  for (std::size_t i = 0; i < numLenParams; ++i) {
    TypeParameterValue v = addendum.LenParameterValue(i);
    hash ^= std::hash<TypeParameterValue>{}(v) +
      0x9e3779b9 + (hash << 6) + (hash >> 2);
  }
  return hash;
}
```

The hash table starts with 31 buckets (currently tunable at build time using
`-DFLANG_RT_PDT_CACHE_INITIAL_BUCKET_CNT`), each holding a linked list of
`CacheEntry` nodes, and the bucket count doubles when the average load
reaches 2 entries per bucket (again, currently tunable at build time using the
define `-DFLANG_RT_PDT_CACHE_MAX_LOAD_FACTOR`).

Bucket arrays are allocated with `std::calloc` while linked-list entries
(`CacheEntry` nodes) use the runtime's `New<T>` / `FreeMemory` allocator.
The concrete `DerivedType` objects themselves are allocated with `std::calloc`,
not `New<T>`.

Hash collisions are theoretically possible but extremely unlikely given a
64-bit hash space. Note that `Find()` compares only hash values, and does not
verify the full key (`genericType` pointer + LEN values) after a match.
A collision between two distinct `(genericType, len_values)` combinations would
therefore silently return the wrong concrete type - a correctness bug,
not merely a performance issue. If collision resistance becomes a concern,
a full-key equality check could be added to `Find()`.

Concrete types are never freed. The runtime currently assumes single-threaded
allocation; if concurrent PDT allocation becomes a requirement, the existing
`Lock` class from `lock.h` could be used.


## Device Compilation

GPU device code cannot use the host-side cache (which relies on `malloc` and
dynamic data structures). When `RT_DEVICE_COMPILATION` is defined,
`GetConcreteType` currently provides a pre-implementation stub that passes
through types without LEN parameters or already-concrete types, and crashes for
actual LEN instantiation requests:

```cpp
#ifdef RT_DEVICE_COMPILATION
RT_API_ATTRS const DerivedType *GetConcreteType(...) {
  if ((numLenParams == 0) || (uninst != nullptr && uninst != &genericType)) {
    return &genericType;
  }
  terminator.Crash("PDT LEN param instantiation not supported in device code");
}
#endif
```


## Integration Points

`GetConcreteType` is called in the following runtime paths:

| File | Function | Purpose |
|------|----------|--------|
| `allocatable.cpp` | `AllocatableInitDerived` | Sets `derivedType_` to concrete on init |
| `allocatable.cpp` | `AllocatableAllocate` | Resolves before allocation |
| `pointer.cpp` | `PointerNullifyDerived` | Sets `derivedType_` to concrete on nullify |
| `pointer.cpp` | `PointerAllocate` | Resolves before allocation |
| `assign.cpp` | `AllocateAssignmentLHS` | Resolves for LHS reallocation |
| `type-info.cpp` | `ResolveDerivedTypeForComponent` | Resolves nested PDT components within `EstablishDescriptor` |

## Version
1.0 - Initial posting
1.1 - Added Recursion description
1.2 - Rearrange sections
