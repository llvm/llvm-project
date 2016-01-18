===========================================
Control Flow Integrity Design Documentation
===========================================

This page documents the design of the :doc:`ControlFlowIntegrity` schemes
supported by Clang.

Forward-Edge CFI for Virtual Calls
==================================

This scheme works by allocating, for each static type used to make a virtual
call, a region of read-only storage in the object file holding a bit vector
that maps onto to the region of storage used for those virtual tables. Each
set bit in the bit vector corresponds to the `address point`_ for a virtual
table compatible with the static type for which the bit vector is being built.

For example, consider the following three C++ classes:

.. code-block:: c++

  struct A {
    virtual void f1();
    virtual void f2();
    virtual void f3();
  };

  struct B : A {
    virtual void f1();
    virtual void f2();
    virtual void f3();
  };

  struct C : A {
    virtual void f1();
    virtual void f2();
    virtual void f3();
  };

The scheme will cause the virtual tables for A, B and C to be laid out
consecutively:

.. csv-table:: Virtual Table Layout for A, B, C
  :header: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14

  A::offset-to-top, &A::rtti, &A::f1, &A::f2, &A::f3, B::offset-to-top, &B::rtti, &B::f1, &B::f2, &B::f3, C::offset-to-top, &C::rtti, &C::f1, &C::f2, &C::f3

The bit vector for static types A, B and C will look like this:

.. csv-table:: Bit Vectors for A, B, C
  :header: Class, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14

  A, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0
  B, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0
  C, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0

Bit vectors are represented in the object file as byte arrays. By loading
from indexed offsets into the byte array and applying a mask, a program can
test bits from the bit set with a relatively short instruction sequence. Bit
vectors may overlap so long as they use different bits. For the full details,
see the `ByteArrayBuilder`_ class.

In this case, assuming A is laid out at offset 0 in bit 0, B at offset 0 in
bit 1 and C at offset 0 in bit 2, the byte array would look like this:

.. code-block:: c++

  char bits[] = { 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 0, 5, 0, 0 };

To emit a virtual call, the compiler will assemble code that checks that
the object's virtual table pointer is in-bounds and aligned and that the
relevant bit is set in the bit vector.

For example on x86 a typical virtual call may look like this:

.. code-block:: none

  ca7fbb:       48 8b 0f                mov    (%rdi),%rcx
  ca7fbe:       48 8d 15 c3 42 fb 07    lea    0x7fb42c3(%rip),%rdx
  ca7fc5:       48 89 c8                mov    %rcx,%rax
  ca7fc8:       48 29 d0                sub    %rdx,%rax
  ca7fcb:       48 c1 c0 3d             rol    $0x3d,%rax
  ca7fcf:       48 3d 7f 01 00 00       cmp    $0x17f,%rax
  ca7fd5:       0f 87 36 05 00 00       ja     ca8511
  ca7fdb:       48 8d 15 c0 0b f7 06    lea    0x6f70bc0(%rip),%rdx
  ca7fe2:       f6 04 10 10             testb  $0x10,(%rax,%rdx,1)
  ca7fe6:       0f 84 25 05 00 00       je     ca8511
  ca7fec:       ff 91 98 00 00 00       callq  *0x98(%rcx)
    [...]
  ca8511:       0f 0b                   ud2

The compiler relies on co-operation from the linker in order to assemble
the bit vectors for the whole program. It currently does this using LLVM's
`bit sets`_ mechanism together with link-time optimization.

.. _address point: https://mentorembedded.github.io/cxx-abi/abi.html#vtable-general
.. _bit sets: http://llvm.org/docs/BitSets.html
.. _ByteArrayBuilder: http://llvm.org/docs/doxygen/html/structllvm_1_1ByteArrayBuilder.html

Optimizations
-------------

The scheme as described above is the fully general variant of the scheme.
Most of the time we are able to apply one or more of the following
optimizations to improve binary size or performance.

In fact, if you try the above example with the current version of the
compiler, you will probably find that it will not use the described virtual
table layout or machine instructions. Some of the optimizations we are about
to introduce cause the compiler to use a different layout or a different
sequence of machine instructions.

Stripping Leading/Trailing Zeros in Bit Vectors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If a bit vector contains leading or trailing zeros, we can strip them from
the vector. The compiler will emit code to check if the pointer is in range
of the region covered by ones, and perform the bit vector check using a
truncated version of the bit vector. For example, the bit vectors for our
example class hierarchy will be emitted like this:

.. csv-table:: Bit Vectors for A, B, C
  :header: Class, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14

  A,  ,  , 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,  ,  
  B,  ,  ,  ,  ,  ,  ,  , 1,  ,  ,  ,  ,  ,  ,  
  C,  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  , 1,  ,  

Short Inline Bit Vectors
~~~~~~~~~~~~~~~~~~~~~~~~

If the vector is sufficiently short, we can represent it as an inline constant
on x86. This saves us a few instructions when reading the correct element
of the bit vector.

If the bit vector fits in 32 bits, the code looks like this:

.. code-block:: none

     dc2:       48 8b 03                mov    (%rbx),%rax
     dc5:       48 8d 15 14 1e 00 00    lea    0x1e14(%rip),%rdx
     dcc:       48 89 c1                mov    %rax,%rcx
     dcf:       48 29 d1                sub    %rdx,%rcx
     dd2:       48 c1 c1 3d             rol    $0x3d,%rcx
     dd6:       48 83 f9 03             cmp    $0x3,%rcx
     dda:       77 2f                   ja     e0b <main+0x9b>
     ddc:       ba 09 00 00 00          mov    $0x9,%edx
     de1:       0f a3 ca                bt     %ecx,%edx
     de4:       73 25                   jae    e0b <main+0x9b>
     de6:       48 89 df                mov    %rbx,%rdi
     de9:       ff 10                   callq  *(%rax)
    [...]
     e0b:       0f 0b                   ud2    

Or if the bit vector fits in 64 bits:

.. code-block:: none

    11a6:       48 8b 03                mov    (%rbx),%rax
    11a9:       48 8d 15 d0 28 00 00    lea    0x28d0(%rip),%rdx
    11b0:       48 89 c1                mov    %rax,%rcx
    11b3:       48 29 d1                sub    %rdx,%rcx
    11b6:       48 c1 c1 3d             rol    $0x3d,%rcx
    11ba:       48 83 f9 2a             cmp    $0x2a,%rcx
    11be:       77 35                   ja     11f5 <main+0xb5>
    11c0:       48 ba 09 00 00 00 00    movabs $0x40000000009,%rdx
    11c7:       04 00 00 
    11ca:       48 0f a3 ca             bt     %rcx,%rdx
    11ce:       73 25                   jae    11f5 <main+0xb5>
    11d0:       48 89 df                mov    %rbx,%rdi
    11d3:       ff 10                   callq  *(%rax)
    [...]
    11f5:       0f 0b                   ud2    

If the bit vector consists of a single bit, there is only one possible
virtual table, and the check can consist of a single equality comparison:

.. code-block:: none

     9a2:   48 8b 03                mov    (%rbx),%rax
     9a5:   48 8d 0d a4 13 00 00    lea    0x13a4(%rip),%rcx
     9ac:   48 39 c8                cmp    %rcx,%rax
     9af:   75 25                   jne    9d6 <main+0x86>
     9b1:   48 89 df                mov    %rbx,%rdi
     9b4:   ff 10                   callq  *(%rax)
     [...]
     9d6:   0f 0b                   ud2

Virtual Table Layout
~~~~~~~~~~~~~~~~~~~~

The compiler lays out classes of disjoint hierarchies in separate regions
of the object file. At worst, bit vectors in disjoint hierarchies only
need to cover their disjoint hierarchy. But the closer that classes in
sub-hierarchies are laid out to each other, the smaller the bit vectors for
those sub-hierarchies need to be (see "Stripping Leading/Trailing Zeros in Bit
Vectors" above). The `GlobalLayoutBuilder`_ class is responsible for laying
out the globals efficiently to minimize the sizes of the underlying bitsets.

.. _GlobalLayoutBuilder: http://llvm.org/viewvc/llvm-project/llvm/trunk/include/llvm/Transforms/IPO/LowerBitSets.h?view=markup

Alignment
~~~~~~~~~

If all gaps between address points in a particular bit vector are multiples
of powers of 2, the compiler can compress the bit vector by strengthening
the alignment requirements of the virtual table pointer. For example, given
this class hierarchy:

.. code-block:: c++

  struct A {
    virtual void f1();
    virtual void f2();
  };

  struct B : A {
    virtual void f1();
    virtual void f2();
    virtual void f3();
    virtual void f4();
    virtual void f5();
    virtual void f6();
  };

  struct C : A {
    virtual void f1();
    virtual void f2();
  };

The virtual tables will be laid out like this:

.. csv-table:: Virtual Table Layout for A, B, C
  :header: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15

  A::offset-to-top, &A::rtti, &A::f1, &A::f2, B::offset-to-top, &B::rtti, &B::f1, &B::f2, &B::f3, &B::f4, &B::f5, &B::f6, C::offset-to-top, &C::rtti, &C::f1, &C::f2

Notice that each address point for A is separated by 4 words. This lets us
emit a compressed bit vector for A that looks like this:

.. csv-table::
  :header: 2, 6, 10, 14

  1, 1, 0, 1

At call sites, the compiler will strengthen the alignment requirements by
using a different rotate count. For example, on a 64-bit machine where the
address points are 4-word aligned (as in A from our example), the ``rol``
instruction may look like this:

.. code-block:: none

     dd2:       48 c1 c1 3b             rol    $0x3b,%rcx

Padding to Powers of 2
~~~~~~~~~~~~~~~~~~~~~~

Of course, this alignment scheme works best if the address points are
in fact aligned correctly. To make this more likely to happen, we insert
padding between virtual tables that in many cases aligns address points to
a power of 2. Specifically, our padding aligns virtual tables to the next
highest power of 2 bytes; because address points for specific base classes
normally appear at fixed offsets within the virtual table, this normally
has the effect of aligning the address points as well.

This scheme introduces tradeoffs between decreased space overhead for
instructions and bit vectors and increased overhead in the form of padding. We
therefore limit the amount of padding so that we align to no more than 128
bytes. This number was found experimentally to provide a good tradeoff.

Eliminating Bit Vector Checks for All-Ones Bit Vectors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the bit vector is all ones, the bit vector check is redundant; we simply
need to check that the address is in range and well aligned. This is more
likely to occur if the virtual tables are padded.

Forward-Edge CFI for Indirect Function Calls
============================================

Sorry, no documentation yet, but see the comments at the top of
``LowerBitSets::buildBitSetsFromFunctions`` in `LowerBitSets.cpp`_.

.. _LowerBitSets.cpp: http://llvm.org/klaus/llvm/blob/master/lib/Transforms/IPO/LowerBitSets.cpp
