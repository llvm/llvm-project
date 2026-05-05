=======================
Typed Memory Operations
=======================

.. contents::
   :local:

Introduction
------------

Typed memory operations provide a mechanism to support type isolating allocators
even in type free C allocation APIs. There are four basic parts to this

* An attribute used to indicate that a function is a typed memory operation, e.g.
  an allocation function that is supported by a type isolating allocator.
* An inference step to infer the actual type of the data the allocation is
  expected to contain.
* A type encoding mechanism to compress the inferred type into a representation
  that the allocator can use. The current implementation only supports a single
  model: the `type_descriptor`
* The code generation phase which re-writes the call as written in code, into a
  call to a specified type aware version of the target function and place the
  additional type parameter.

This allows even system allocation functions (like malloc, and similar) to be
annotated in a way that permits automatic, transparent, and code-change free (in
the general case) migration of existing C code to type isolating allocation.

Using function agnostic attributes results in the functionality being generally
available to user provided custom allocators as well: fully custom allocators
can introduce type aware interfaces without requiring manual user adoption, and
code that uses wrappers around allocators can also add type aware interfaces
that can forward the type information to their underlying allocators, without
depending on implicit knowledge by the compiler.


``typed_memory_operation`` attribute
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The core developer facing component of TMO is the `typed_memory_operation`
function attribute. This attribute specifies the expression that should be used
to infer the allocation type, and the function that should be used as the
re-write target.

The `typed_memory_operation` attribute has two parameters - the first is the
1-based index of the size argument from which the allocation type should be
inferred. The second parameter is the name of the type aware allocation function
that will be called instead.

The attribute can then be used as follows:

.. code-block:: c

  void *typed_malloc(size_t size, type_descriptor_t type);
  void *malloc(size_t size) __attribute__((typed_memory_operation(1, typed_malloc)))

This will result in the allocation of a call to `malloc` having a type inferred
from the `size` argument, and the call being rewritten to `typed_malloc`. e.g.

.. code-block:: c

  malloc(sizeof(T));

Gets emitted as a call to `typed_malloc`

.. code-block:: c

  typed_malloc(sizeof(T), [[type descriptor]]);

When re-writing a type aware allocation the type information is passed as an
inserted parameter following the inference target expression.

.. code-block:: c

  void *typed_custom_alloc(size_t size, context_t* ctx, type_descriptor_t type);
  void *custom_alloc(size_t size, context_t* ctx)
    __attribute__((typed_memory_operation(1, typed_custom_alloc)))
  // ...
  custom_alloc(sizeof(T), &context);

results in a call to `typed_custom_alloc`

.. code-block:: c

  typed_custom_alloc(sizeof(T), [[type descriptor]], ctx);

This design is intended to allow future support for functions with multiple
inference targets.

``__builtin_tmo_get_type_descriptor``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In addition to the primary attribute the `__builtin_tmo_get_type_descriptor`
provides a mechanism to get the type descriptor for a known type. This makes it
possible for developers to avoid trying to construct an expression that will
infer the correct type when they already have full knowledge (as can occur with
allocator macros, or C++ type aware allocators).

.. code-block:: c

  __builtin_tmo_get_type_descriptor(type)

Computes the type descriptor for the given type such that the encoded type
matches the type descriptor that would be produced by the inference pass.

Type inference
--------------

Type inference is the most critical component of this feature as it is
responsible for the core functional requirement: providing the allocation type.

The inference algorithm should not be considered stable: the intention of this
process is to provide the most accurately inferred type information that is
possible, and so will be improved over time.

The core inference algorithm is focused on the specified operation parameter,
and currently matches all common allocation size idioms:

  * Single objects and arrays, e.g. `sizeof(T)`, `sizeof(T)` * expression
  * Prefixed arrays, both variations on `sizeof(Header) + sizeof(Tail) * expression`
    and non-explicit tail allocations for types with flexible array members
  * tuple allocations, like `sizeof(T) + sizeof(U)`

The inference process also follows variable references if the assigned value
can be trivially determined.

The inference process also tracks casts and similar operations performed over
the returned allocation to provide further information about the likely object
type that is used either as a fallback if no type can be inferred by an
allocation, or to fine tune the semantics of the inferred type.

If the inference cannot determine an exact allocation structure it reduces the
precision to an unspecified combination of all types referenced in the expression.

Type descriptor format
~~~~~~~~~~~~~~~~~~~~~~

The type descriptor is a 64bit value containing a summary of the semantic
features of the inferred type, and a hash of the structural identity of the
type.

Identity
~~~~~~~~

Typed memory operations work in terms of a type identity that is independent of C.
Using something like a C type name, or a description of the type as specified,
results in significant increases in code size while also producing large numbers
of structurally identical types.

Instead a structural identity is constructed by converting each type into a
sequence of atoms representing the actual data stored in each atom of the type
when realised in memory. Overlapping atoms (as occur in unions) have their
properties unified.

The result of this process is that multiple types or type definitions that have
structurally equivalent types are reduced to the same identity, permitting an
allocator to unify their allocations. This avoids an allocator having to manage
an explosion of type buckets, even when a functionally limited number of unique
types are actually being allocated.

The information being tracked during this process is an approximation of the
content, that is intended to be increased over time to improve both the
unification and the isolation of types.

Summary
~~~~~~~

The remainder of the descriptor is a summary of the semantic properties of the
type[s] being allocated. This is laid out as follows.

.. code-block:: text

  Bits 31–16: Layout semantics
  Bits 15–12: Type flags
  Bits 11–10: Type kind
  Bits  9– 6: Callsite flags
  Bits  5– 2: Reserved
  Bits  1– 0: Version (currently 0)

The exact values and contents of these are specified in `TypedMemoryDescriptorBits.h`.

Reserved and Version are self explanatory, as an overview of the other fields

* Layout semantics is a bitfield to allow the allocator to identify whether the
  type being allocated contains certain kinds of data, e.g. pointers, code
  pointers, whether it is pure data, etc.
* Type flags represents type specific semantics, current examples being the
  existence of polymorphic objects, or unions containing different data types.
* Type kind represents the originating language or allocation class (e.g. a C
  like allocation, new/delete, swift based allocations).
* Callsite flags represent information about the callsite rather than the type
  itself - is the allocation fixed or variable size, array vs non-array, if
  there is an object header.

Inference failure
~~~~~~~~~~~~~~~~~

Type inference in these function calls is still just a heuristic, so can fail.
In the event that the inference process cannot infer any type information the
descriptor that is passed is produced by hashing the source code location of the
allocation site, so that the allocator always has at least a semi-unique
identity to use as a basis of type isolation.

Future work
-----------

Full inferred-type encoding
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The current descriptor does not include the logic that encodes the full
inferred type information. This is a regression to be fixed, but the additional
complexity makes this already large feature larger.

Descriptor abstraction layer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The current descriptor format is fixed and reflects one particular set of design
choices about what information is useful at the runtime level. The RFC discussed
this limitation, and there are other type encoding systems that are valuable in
other contexts. A basic abstraction layer to make these encodings configurable
is an obvious future step.

Explicit return identification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As currently implemented the inference algorithm assumes that all allocation
functions that return a pointer are returning the allocation, and incorporates
that into its inference logic. This works for many cases, but in some cases is
wrong and so the return value should be ignored as it may pollute the inference
process. At the same time it also means that information that could be extracted
from out parameters is lost.

Variable reference inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Currently the logic for performing inference through variable references is
extremely limited: constant initialized variables. There are many trivial cases
this misses as a result of code making sequential variable changes to reduce
code complexity or to make the code easier to read. Most of these cases involve
linear sequences of assignments that should be handled as such.
