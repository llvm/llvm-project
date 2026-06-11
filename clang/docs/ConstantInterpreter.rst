====================
Constant Interpreter
====================

.. contents::
   :local:

Introduction
============

The bytecode interpreter aims to replace the existing AST traversal-based
evaluator in Clang, improving performance on constructs which are executed
inefficiently by the evaluator. The interpreter is activated by passing
``-fexperimental-new-constant-interpreter`` to clang.

Since Clang 23, the bytecode interpreter can also be enabled by default
by passing ``-DCLANG_USE_EXPERIMENTAL_CONST_INTERP=ON`` to cmake. In
that case, it can be deactivated again via
``-fno-experimental-new-constant-interpreter``.



Bytecode Compilation
====================

Bytecode compilation is handled in ``Compiler.h`` for statements
and for expressions. The compiler has two different
backends: one to generate bytecode for functions (``ByteCodeEmitter``) and
one to directly evaluate expressions as they are compiled, without
generating bytecode (``EvalEmitter``). All functions are compiled to
bytecode, while toplevel expressions used in constant contexts are directly
evaluated since the bytecode would never be reused. This mechanism aims to
pave the way towards replacing the evaluator, improving its performance on
functions and loops, while being just as fast on single-use toplevel
expressions.

The interpreter relies on stack-based, strongly-typed opcodes. The glue
logic between the code generator, along with the enumeration and
description of opcodes, can be found in ``Opcodes.td``. The opcodes are
implemented as generic template methods in ``Interp.h`` and instantiated
with the relevant primitive types by the interpreter loop or by the
evaluating emitter.

Primitive Types
---------------

* ``PT_Char``

  Signed or unsigned 1-byte character type.

* ``PT_{U|S}int{16|32|64}``

  Signed or unsigned integers of a specific bit width, implemented using
  the ``Integral`` type. All of these take up 24 bytes each since they
  need to be able to represent a pointer that has been casted to an integer.


* ``PT_IntAP{S}``

  Signed or unsigned integers of an arbitrary, but fixed width used to
  implement integral types which are required by the target, but are not
  supported by the host. Under the hood, they rely on ``APInt``. The
  ``Integral`` specialisation for these types is required by opcodes to
  share an implementation with fixed integrals.

* ``PT_Bool``

  Representation for boolean types, essentially a 1-bit unsigned
  ``Integral``.

* ``PT_Float``

  Arbitrary, but fixed precision floating point numbers. Could be
  specialised in the future similarly to integers in order to improve
  floating point performance.

* ``PT_Ptr``

  Pointer type, defined in ``"Pointer.h"``. The most common type of
  pointer is a "BlockPointer", which points to an ``interp::Block``.
  But other pointer types exist, such as typeid pointers or
  integral pointers.

* ``PT_MemberPtr``

  Member pointer type, can also be a null member pointer. Defined
  in ``"MemberPointer.h"``

Composite types
---------------

The interpreter distinguishes two kinds of composite types: arrays and
records (structs and classes). Unions are represented as records, except
at most a single field can be marked as active. The contents of inactive
fields are kept until they are reactivated and overwritten.
Complex numbers (``_Complex``) and vectors
(``__attribute((vector_size(16)))``) are treated as arrays.


Bytecode Execution
==================

Bytecode is executed using a stack-based interpreter. The execution
context consists of an ``InterpStack``, along with a chain of
``InterpFrame`` objects storing the call frames. Frames are built by
call instructions and destroyed by return instructions. They perform
one allocation to reserve space for all locals in a single block.
These objects store all the required information to emit stack traces
whenever evaluation fails.

Memory Organisation
===================

Memory management in the interpreter relies on 3 data structures: ``Block``
objects which store the data and associated inline metadata, ``Pointer``
objects which refer to or into blocks, and ``Descriptor`` structures which
describe blocks and subobjects nested inside blocks.

Blocks
------

Blocks contain data interleaved with metadata. They are allocated either
statically in the code generator (globals, static members, dummy parameter
values etc.) or dynamically in the interpreter, when creating the frame
containing the local variables of a function. Blocks are associated with a
descriptor that characterises the entire allocation, along with a few
additional attributes:

* ``IsStatic`` indicates whether the block has static duration in the
  interpreter, i.e. it is not a local in a frame.

* ``DeclID`` identifies each global declaration (it is set to an invalid
  and irrelevant value for locals) in order to prevent illegal writes and
  reads involving globals and temporaries with static storage duration.

Static blocks are never deallocated, but local ones might be deallocated
even when there are live pointers to them. Pointers are only valid as
long as the blocks they point to are valid, so a block with pointers to
it whose lifetime ends is kept alive until all pointers to it go out of
scope. Since the frame is destroyed on function exit, such blocks are
turned into a ``DeadBlock`` and copied to storage managed by the
interpreter itself, not the frame. Reads and writes to these blocks are
illegal and cause an appropriate diagnostic to be emitted. When the last
pointer goes out of scope, dead blocks are also deallocated.

The lifetime of blocks is managed through 2 methods stored in the
descriptor of the block:

* **CtorFn**: initializes the metadata which is stored in the block,
  alongside actual data. Invokes the default constructors of objects
  which are not trivial (``Pointer``, ``Floating``, etc.)

* **DtorFn**: invokes the destructors of non-trivial objects.

Blocks track all the pointers into them through an intrusive
doubly-linked list, required to adjust and invalidate all pointers when
transforming a block into a dead block. If the lifetime of an object ends,
all pointers to it are invalidated, emitting the appropriate diagnostics when
dereferenced.


Records are laid out identically to arrays of composites: each field and base
class is preceded by an inline descriptor. The ``InlineDescriptor`` saves
information about the initialization state, constness, mutability, lifetime,
etc. of a field.

Consider this struct:

.. code-block:: c++

    struct S {
        char c;
    };
    constexpr S s{12};

When allocating space for ``s``, we allocate 24 bytes (not counting the ``Descriptor``
instances we created for the ``Record`` and the field). The field ``c`` needs 8 bytes,
since we align to pointer size (this example uses a 64 bit system). The
``InlineDescriptor`` preceding the field data uses up the remaining 16 bytes.

Inline descriptors are filled in by the ``CtorFn`` of blocks, which leaves storage
in an uninitialised, but valid state.

Descriptors
-----------

Descriptors are generated at bytecode compilation time and contain information
required to determine if a particular memory access is allowed in constexpr.
They also carry all the information required to emit a diagnostic involving
a memory access, such as the declaration which originates the block.
Currently there is a single kind of descriptor encoding information for all
block types.

Pointers
--------

Pointers, implemented in ``Pointer.h`` are represented as a tagged union.

 * **BlockPointer**: used to reference memory allocated and managed by the
   interpreter, being the only pointer kind which allows dereferencing in the
   interpreter
 * **TypeIDPointer**: tracks information for the opaque type returned by
   ``typeid``
 * **IntegralPointer**: a pointer formed from an integer,
   think ``(int*)123``.
 * **FunctionPointer**: a pointer to a function.

Besides the previously mentioned union, a number of other pointer-like types
have their own type:

 * **FunctionPointer** tracks functions.
 * **MemberPointer** tracks C++ object members

BlockPointer
~~~~~~~~~~~~

Block pointers track a ``Pointee``, the block to which they point, along
with a ``Base`` and an ``Offset``. The base identifies the innermost field,
while the offset points to an array element relative to the base (including
one-past-end pointers). The offset identifies the array element or field
which is referenced, while the base points to the outer object or array which
contains the field. These two fields allow all pointers to be uniquely
identified, disambiguated and characterised.

As an example, consider the following structure:

.. code-block:: c

    struct A {
        struct B {
            int x;
            int y;
        } b;
        struct C {
            int a;
            int b;
        } c[2];
        int z;
    };
    constexpr A a;

On the target, ``&a`` and ``&a.b.x`` are equal. So are ``&a.c[0]`` and
``&a.c[0].a``. In the interpreter, all these pointers must be
distinguished since they are all allowed to address a distinct range of
memory.

In the interpreter, the object would require 240 bytes of storage and
would have its fields interleaved with metadata. The pointers which can
be derived to the object are illustrated in the following diagram:

::

      0   16  32  40  56  64  80  96  112 120 136 144 160 176 184 200 208 224 240
  +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
  + B | D | D | x | D | y | D | D | D | a | D | b | D | D | a | D | b | D | z |
  +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
      ^   ^   ^       ^       ^   ^   ^       ^       ^   ^       ^       ^
      |   |   |       |       |   |   |   &a.c[0].b   |   |   &a.c[1].b   |
      a   |&a.b.x   &a.y    &a.c  |&a.c[0].a          |&a.c[1].a          |
        &a.b                   &a.c[0]            &a.c[1]               &a.z

The ``Base`` offset of all pointers points to the start of a field or
an array and is preceded by an inline descriptor (unless ``Base`` is
zero, pointing to the root). All the relevant attributes can be read
from either the inline descriptor or the descriptor of the block.


Array elements are identified by the ``Offset`` field of pointers,
pointing to past the inline descriptors for composites and before
the actual data in the case of primitive arrays. The ``Offset``
points to the offset where primitives can be read from. As an example,
``a.c + 1`` would have the same base as ``a.c`` since it is an element
of ``a.c``, but its offset would point to ``&a.c[1]``. The
array-to-pointer decay operation adjusts a pointer to an array (where
the offset is equal to the base) to a pointer to the first element.

TypeInfoPointer
~~~~~~~~~~~~~~~

``TypeInfoPointer`` tracks two types: the type assigned to
``std::type_info`` and the type which was passed to ``typeinfo``.
It is part of the tagged union in ``Pointer``.



Interpretation
--------------
After bytecode has been generated (or not, for expressions), the bytecode
is then interpreted. The bytecode is stack-based and uses ``InterpStack``
to allocate memory for the produced values.

Here is an example function:

.. code-block:: c++

    constexpr int add(int a, int b) {
      return a + b;
    }
    static_assert(add(1, 2) == 3);

Which generates the following bytecode (this can be produced via ``interp::Function::dump()``):

::

  add 0x7cb97f7e2000
  [...]
  0     GetParamSint32    0
  16    GetParamSint32    1
  32    AddSint32
  40    RetSint32
  48    NoRet

As you can see, all instructions here are type-aware. We're first pushing both parameter values
to the stack. Then the ``Add`` opcode will add them up and push the result to the stack, which
will be returned via the ``Ret`` opcode.


Debugging
---------
Here are a few hints when working on the bytecode interpreter:

* Setting a breakpoint on ``CCEDiag`` and ``FFDiag`` will stop the debugger when a diagnostic
  is emitted.
* If you want to see the bytecode of a function call ``dump()`` on the ``interp::Function``.
* Additionally to the last point, ``interp::Function::dump(CodePtr)`` exists, which you can
  pass the current ``OpPC`` and it will then show you where in the bytecode that opcode is.
* ``interp::Pointer`` has an ``operator<<`` that prints useful information. Try it e.g.
  via ``llvm::errs() << Ptr << '\n';``.
* Printing ``APValue`` instances also works via ``APValue::dump()``.
* If you want to see *everything* that's being evaluated, add debugging output to
  the ``evaluate*`` functions in ``Context.cpp``.
