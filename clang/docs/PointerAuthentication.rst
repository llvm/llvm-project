Pointer Authentication
======================

.. contents::
   :local:

Introduction
------------

Pointer authentication is a technology which offers strong probabilistic
protection against exploiting a broad class of memory bugs to take control of
program execution.  When adopted consistently in a language ABI, it provides
a form of relatively fine-grained control flow integrity (CFI) check that
resists both return-oriented programming (ROP) and jump-oriented programming
(JOP) attacks.

While pointer authentication can be implemented purely in software, direct
hardware support (e.g. as provided by Armv8.3 PAuth) can dramatically improve
performance and code size.  Similarly, while pointer authentication
can be implemented on any architecture, taking advantage of the (typically)
excess addressing range of a target with 64-bit pointers minimizes the impact
on memory performance and can allow interoperation with existing code (by
disabling pointer authentication dynamically).  This document will generally
attempt to present the pointer authentication feature independent of any
hardware implementation or ABI.  Considerations that are
implementation-specific are clearly identified throughout.

Note that there are several different terms in use:

- **Pointer authentication** is a target-independent language technology.

- **PAuth** (sometimes referred to as **PAC**, for Pointer Authentication
  Codes) is an AArch64 architecture extension that provides hardware support
  for pointer authentication.  Additional extensions either modify some of the
  PAuth instruction behavior (notably FPAC), or provide new instruction
  variants (PAuth_LR).

- **Armv8.3** is an AArch64 architecture revision that makes PAuth mandatory.

- **arm64e** is a specific ABI (not yet fully stable) for implementing pointer
  authentication using PAuth on certain Apple operating systems.

This document serves four purposes:

- It describes the basic ideas of pointer authentication.

- It documents several language extensions that are useful on targets using
  pointer authentication.

- It presents a theory of operation for the security mitigation, describing the
  basic requirements for correctness, various weaknesses in the mechanism, and
  ways in which programmers can strengthen its protections (including
  recommendations for language implementors).

- It documents the stable ABI of the C, C++, and Objective-C languages on arm64e
  platforms.


Basic concepts
--------------

The simple address of an object or function is a **raw pointer**.  A raw
pointer can be **signed** to produce a **signed pointer**.  A signed pointer
can be then **authenticated** in order to verify that it was **validly signed**
and extract the original raw pointer.  These terms reflect the most likely
implementation technique: computing and storing a cryptographic signature along
with the pointer.

An **abstract signing key** is a name which refers to a secret key which is
used to sign and authenticate pointers.  The concrete key value for a
particular name is consistent throughout a process.

A **discriminator** is an arbitrary value used to **diversify** signed pointers
so that one validly-signed pointer cannot simply be copied over another.
A discriminator is simply opaque data of some implementation-defined size that
is included in the signature as a salt (see `Discriminators`_ for details.)

Nearly all aspects of pointer authentication use just these two primary
operations:

- ``sign(raw_pointer, key, discriminator)`` produces a signed pointer given
  a raw pointer, an abstract signing key, and a discriminator.

- ``auth(signed_pointer, key, discriminator)`` produces a raw pointer given
  a signed pointer, an abstract signing key, and a discriminator.

``auth(sign(raw_pointer, key, discriminator), key, discriminator)`` must
succeed and produce ``raw_pointer``.  ``auth`` applied to a value that was
ultimately produced in any other way is expected to fail, which halts the
program either:

- immediately, on implementations that enforce ``auth`` success (e.g., when
  using compiler-generated ``auth`` failure checks, or Armv8.3 with the FPAC
  extension), or

- when the resulting pointer value is used, on implementations that don't.

However, regardless of the implementation's handling of ``auth`` failures, it
is permitted for ``auth`` to fail to detect that a signed pointer was not
produced in this way, in which case it may return anything; this is what makes
pointer authentication a probabilistic mitigation rather than a perfect one.

There are two secondary operations which are required only to implement certain
intrinsics in ``<ptrauth.h>``:

- ``strip(signed_pointer, key)`` produces a raw pointer given a signed pointer
  and a key without verifying its validity, unlike ``auth``.  This is useful
  for certain kinds of tooling, such as crash backtraces; it should generally
  not be used in the basic language ABI except in very careful ways.

- ``sign_generic(value)`` produces a cryptographic signature for arbitrary
  data, not necessarily a pointer.  This is useful for efficiently verifying
  that non-pointer data has not been tampered with.

Whenever any of these operations is called for, the key value must be known
statically.  This is because the layout of a signed pointer may vary according
to the signing key.  (For example, in Armv8.3, the layout of a signed pointer
depends on whether Top Byte Ignore (TBI) is enabled, which can be set
independently for I and D keys.)

.. admonition:: Note for API designers and language implementors

  These are the *primitive* operations of pointer authentication, provided for
  clarity of description.  They are not suitable either as high-level
  interfaces or as primitives in a compiler IR because they expose raw
  pointers.  Raw pointers require special attention in the language
  implementation to avoid the accidental creation of exploitable code
  sequences; see the section on `Attackable code sequences`_.

The following details are all implementation-defined:

- the nature of a signed pointer
- the size of a discriminator
- the number and nature of the signing keys
- the implementation of the ``sign``, ``auth``, ``strip``, and ``sign_generic``
  operations

While the use of the terms "sign" and "signed pointer" suggest the use of
a cryptographic signature, other implementations may be possible.  See
`Alternative implementations`_ for an exploration of implementation options.

.. admonition:: Implementation example: Armv8.3

  Readers may find it helpful to know how these terms map to Armv8.3 PAuth:

  - A signed pointer is a pointer with a signature stored in the
    otherwise-unused high bits.  The kernel configures the address width based
    on the system's addressing needs, and enables TBI for I or D keys as
    needed.  The bits above the address bits and below the TBI bits (if
    enabled) are unused.  The signature width then depends on this addressing
    configuration.

  - A discriminator is a 64-bit integer.  Constant discriminators are 16-bit
    integers.  Blending a constant discriminator into an address consists of
    replacing the top 16 bits of the pointer containing the address with the
    constant.  Pointers used for blending purposes should only have address
    bits, since higher bits will be at least partially overwritten with the
    constant discriminator.

  - There are five 128-bit signing-key registers, each of which can only be
    directly read or set by privileged code.  Of these, four are used for
    signing pointers, and the fifth is used only for ``sign_generic``.  The key
    data is simply a pepper added to the hash, not an encryption key, and so
    can be initialized using random data.

  - ``sign`` computes a cryptographic hash of the pointer, discriminator, and
    signing key, and stores it in the high bits as the signature. ``auth``
    removes the signature, computes the same hash, and compares the result with
    the stored signature.  ``strip`` removes the signature without
    authenticating it.  The ``aut`` instructions in the baseline Armv8.3 PAuth
    feature do not guarantee to trap on authentication failure; instead, they
    simply corrupt the pointer so that later uses will likely trap. Unless the
    "later use" follows immediately and cannot be recovered from (e.g. with a
    signal handler), this does not provide adequate protection against
    `authentication oracles`_, so implementations must emit additional
    instructions to force an immediate trap. This is unnecessary if the
    processor provides the optional ``FPAC`` extension, which guarantees an
    immediate trap.

  - ``sign_generic`` corresponds to the ``pacga`` instruction, which takes two
    64-bit values and produces a 64-bit cryptographic hash. Implementations of
    this instruction are not required to produce meaningful data in all bits of
    the result.

Discriminators
~~~~~~~~~~~~~~

A discriminator is arbitrary extra data which alters the signature calculated
for a pointer.  When two pointers are signed differently --- either with
different keys or with different discriminators --- an attacker cannot simply
replace one pointer with the other.

To use standard cryptographic terminology, a discriminator acts as a
`salt <https://en.wikipedia.org/wiki/Salt_(cryptography)>`_ in the signing of a
pointer, and the key data acts as a
`pepper <https://en.wikipedia.org/wiki/Pepper_(cryptography)>`_.  That is,
both the discriminator and key data are ultimately just added as inputs to the
signing algorithm along with the pointer, but they serve significantly
different roles.  The key data is a common secret added to every signature,
whereas the discriminator is a value that can be derived from
the context in which a specific pointer is signed.  However, unlike a password
salt, it's important that discriminators be *independently* derived from the
circumstances of the signing; they should never simply be stored alongside
a pointer.  Discriminators are then re-derived in authentication operations.

The intrinsic interface in ``<ptrauth.h>`` allows an arbitrary discriminator
value to be provided, but can only be used when running normal code.  The
discriminators used by language ABIs must be restricted to make it feasible for
the loader to sign pointers stored in global memory without needing excessive
amounts of metadata.  Under these restrictions, a discriminator may consist of
either or both of the following:

- The address at which the pointer is stored in memory.  A pointer signed with
  a discriminator which incorporates its storage address is said to have
  **address diversity**.  In general, using address diversity means that
  a pointer cannot be reliably copied by an attacker to or from a different
  memory location.  However, an attacker may still be able to attack a larger
  call sequence if they can alter the address through which the pointer is
  accessed.  Furthermore, some situations cannot use address diversity because
  of language or other restrictions.

- A constant integer, called a **constant discriminator**. A pointer signed
  with a non-zero constant discriminator is said to have **constant
  diversity**.  If the discriminator is specific to a single declaration, it is
  said to have **declaration diversity**; if the discriminator is specific to
  a type of value, it is said to have **type diversity**.  For example, C++
  v-tables on arm64e sign their component functions using a hash of their
  method names and signatures, which provides declaration diversity; similarly,
  C++ member function pointers sign their invocation functions using a hash of
  the member pointer type, which provides type diversity.

The implementation may need to restrict constant discriminators to be
significantly smaller than the full size of a discriminator.  For example, on
arm64e, constant discriminators are only 16-bit values.  This is believed to
not significantly weaken the mitigation, since collisions remain uncommon.

The algorithm for blending a constant discriminator with a storage address is
implementation-defined.

.. _Signing schemas:

Signing schemas
~~~~~~~~~~~~~~~

Correct use of pointer authentication requires the signing code and the
authenticating code to agree about the **signing schema** for the pointer:

- the abstract signing key with which the pointer should be signed and
- an algorithm for computing the discriminator.

As described in the section above on `Discriminators`_, in most situations, the
discriminator is produced by taking a constant discriminator and optionally
blending it with the storage address of the pointer.  In these situations, the
signing schema breaks down even more simply:

- the abstract signing key,
- a constant discriminator, and
- whether to use address diversity.

It is important that the signing schema be independently derived at all signing
and authentication sites.  Preferably, the schema should be hard-coded
everywhere it is needed, but at the very least, it must not be derived by
inspecting information stored along with the pointer.  See the section on
`Attacks on pointer authentication`_ for more information.


Language features
-----------------

There are three levels of the pointer authentication language feature:

- The language implementation automatically signs and authenticates function
  pointers (and certain data pointers) across a variety of standard situations,
  including return addresses, function pointers, and C++ virtual functions. The
  intent is for all pointers to code in program memory to be signed in some way
  and for all branches to code in program text to authenticate those
  signatures. In addition to the code pointers themselves, we also use pointer
  authentication to protect data values that directly or indirectly influence
  control flow or program integrity, or can provide attackers with some other
  powerful program compromise.

- The language also provides extensions to override the default rules used by
  the language implementation.  For example, the ``__ptrauth`` type qualifier
  can be used to change how pointers or pointer sized integers are signed when
  they are stored in a particular variable or field; this provides much stronger
  protection than is guaranteed by the default rules for C function and data
  pointers.

- Finally, the language provides the ``<ptrauth.h>`` intrinsic interface for
  manually signing and authenticating pointers in code.  These can be used in
  circumstances where very specific behavior is required.

Language implementation
~~~~~~~~~~~~~~~~~~~~~~~

For the most part, pointer authentication is an unobserved detail of the
implementation of the programming language.  Any element of the language
implementation that would perform an indirect branch to a pointer is implicitly
altered so that the pointer is signed when first constructed and authenticated
when the branch is performed.  This includes:

- indirect-call features in the programming language, such as C function
  pointers, C++ virtual functions, C++ member function pointers, the "blocks"
  C extension, and so on;

- returning from a function, no matter how it is called; and

- indirect calls introduced by the implementation, such as branches through the
  global offset table (GOT) used to implement direct calls to functions defined
  outside of the current shared object.

For more information about this, see the `Language ABI`_ section.

However, some aspects of the implementation are observable by the programmer or
otherwise require special notice.

C data pointers
^^^^^^^^^^^^^^^

The current implementation in Clang does not sign pointers to ordinary data by
default. For a partial explanation of the reasoning behind this, see the
`Theory of Operation`_ section.

A specific data pointer which is more security-sensitive than most can be
signed using the `__ptrauth qualifier`_ or using the ``<ptrauth.h>``
intrinsics.

C function pointers
^^^^^^^^^^^^^^^^^^^

The C standard imposes restrictions on the representation and semantics of
function pointer types which make it difficult to achieve satisfactory
signature diversity in the default language rules.  See `Attacks on pointer
authentication`_ for more information about signature diversity.  Programmers
should strongly consider using the ``__ptrauth`` qualifier to improve the
protections for important function pointers, such as the components of of
a hand-rolled "v-table"; see the section on the `__ptrauth qualifier`_ for
details.

The value of a pointer to a C function includes a signature, even when the
value is cast to a non-function-pointer type like ``void*`` or ``intptr_t``. On
implementations that use high bits to store the signature, this means that
relational comparisons and hashes will vary according to the exact signature
value, which is likely to change between executions of a program.  In some
implementations, it may also vary based on the exact function pointer type.

Null pointers
^^^^^^^^^^^^^

In principle, an implementation could derive the signed null pointer value
simply by applying the standard signing algorithm to the raw null pointer
value. However, for likely signing algorithms, this would mean that the signed
null pointer value would no longer be statically known, which would have many
negative consequences.  For one, it would become substantially more expensive
to emit null pointer values or to perform null-pointer checks.  For another,
the pervasive (even if technically unportable) assumption that null pointers
are bitwise zero would be invalidated, making it substantially more difficult
to adopt pointer authentication, as well as weakening common optimizations for
zero-initialized memory such as the use of ``.bzz`` sections.  Therefore it is
beneficial to treat null pointers specially by giving them their usual
representation.  On AArch64, this requires additional code when working with
possibly-null pointers, such as when copying a pointer field that has been
signed with address diversity.

While this representation of nulls is the safest option for the general case,
there are some situations in which a null pointer may have important semantic
or security impact. For that purpose Clang has the concept of a pointer
authentication schema that signs and authenticates null values.

Return addresses
^^^^^^^^^^^^^^^^

The current implementation in Clang implicitly signs the return addresses in
function calls.  While the value of the return address is technically an
implementation detail of a function, there are some important libraries and
development tools which rely on manually walking the chain of stack frames.
These tools must be updated to correctly account for pointer authentication,
either by stripping signatures (if security is not important for the tool, e.g.
if it is capturing a stack trace during a crash) or properly authenticating
them.  More information about how these values are signed is available in the
`Language ABI`_ section.

C++ virtual functions
^^^^^^^^^^^^^^^^^^^^^

The current implementation in Clang signs virtual function pointers with
a discriminator derived from the full signature of the overridden method,
including the method name and parameter types.  It is possible to write C++
code that relies on v-table layout remaining constant despite changes to
a method signature; for example, a parameter might be a ``typedef`` that
resolves to a different type based on a build setting.  Such code violates
C++'s One Definition Rule (ODR), but that violation is not normally detected;
however, pointer authentication will detect it.

Language extensions
~~~~~~~~~~~~~~~~~~~

Feature testing
^^^^^^^^^^^^^^^

Whether the current target uses pointer authentication can be tested for with
a number of different tests.

- ``__PTRAUTH__`` macro is defined if ``<ptrauth.h>`` provides its normal
  interface. This implies support for the pointer authentication intrinsics
  and the ``__ptrauth`` qualifier.

- ``__has_feature(ptrauth_returns)`` is true if the target uses pointer
  authentication to protect return addresses.

- ``__has_feature(ptrauth_calls)`` is true if the target uses pointer
  authentication to protect indirect branches.  On arm64e this implies
  ``__has_feature(ptrauth_returns)``, ``__has_feature(ptrauth_intrinsics)``,
  and the ``__PTRAUTH__`` macro.

- For backwards compatibility purposes ``__has_feature(ptrauth_intrinsics)``
  and ``__has_feature(ptrauth_qualifier)`` are available on arm64e targets.
  These features are synonymous with each other, and are equivalent to testing
  for the ``__PTRAUTH__`` macro definition. Use of these features should be
  restricted to cases where backwards compatibility is required, and should be
  paired with ``defined(__PTRAUTH__)``.


Clang provides several other tests only for historical purposes; for current
purposes they are all equivalent to ``ptrauth_calls``.

``__ptrauth`` qualifier
^^^^^^^^^^^^^^^^^^^^^^^

``__ptrauth(key, address, discriminator)`` is an extended type
qualifier which causes so-qualified objects to hold pointers or pointer sized
integers signed using the specified schema rather than the default schema for
such types.

In the current implementation in Clang, the qualified type must be a C pointer
type, either to a function or to an object, or a pointer sized integer.  It
currently cannot be an Objective-C pointer type, a C++ reference type, or a
block pointer type; these restrictions may be lifted in the future.

The current implementation in Clang is known to not provide adequate safety
guarantees against the creation of `signing oracles`_ when assigning data
pointers to ``__ptrauth``-qualified gl-values.  See the section on `safe
derivation`_ for more information.

The qualifier's operands are as follows:

- ``key`` - an expression evaluating to a key value from ``<ptrauth.h>``; must
  be a constant expression

- ``address`` - whether to use address diversity (1) or not (0); must be
  a constant expression with one of these two values

- ``discriminator`` - a constant discriminator; must be a constant expression

See `Discriminators`_ for more information about discriminators.

Currently the operands must be constant-evaluable even within templates. In the
future this restriction may be lifted to allow value-dependent expressions as
long as they instantiate to a constant expression.

Consistent with the ordinary C/C++ rule for parameters, top-level ``__ptrauth``
qualifiers on a parameter (after parameter type adjustment) are ignored when
deriving the type of the function.  The parameter will be passed using the
default ABI for the unqualified pointer type.

If ``x`` is an object of type ``__ptrauth(key, address, discriminator) T``,
then the signing schema of the value stored in ``x`` is a key of ``key`` and
a discriminator determined as follows:

- if ``address`` is 0, then the discriminator is ``discriminator``;

- if ``address`` is 1 and ``discriminator`` is 0, then the discriminator is
  ``&x``; otherwise

- if ``address`` is 1 and ``discriminator`` is non-zero, then the discriminator
  is ``ptrauth_blend_discriminator(&x, discriminator)``; see
  `ptrauth_blend_discriminator`_.

Non-triviality from address diversity
+++++++++++++++++++++++++++++++++++++

Address diversity must impose additional restrictions in order to allow the
implementation to correctly copy values.  In C++, a type qualified with address
diversity is treated like a class type with non-trivial copy/move constructors
and assignment operators, with the usual effect on containing classes and
unions.  C does not have a standard concept of non-triviality, and so we must
describe the basic rules here, with the intention of imitating the emergent
rules of C++:

- A type may be **non-trivial to copy**.

- A type may also be **illegal to copy**. Types that are illegal to copy are
  always non-trivial to copy.

- A type may also be **address-sensitive**. This includes types that use self
  referencing pointers, data protected by address diversified pointer
  authentication, or other similar concepts.

- A type qualified with a ``ptrauth`` qualifier or implicit authentication
  schema that requires address diversity is non-trivial to copy and
  address-sensitive.

- An array type is illegal to copy, non-trivial to copy, or address-sensitive
  if its element type is illegal to copy, non-trivial to copy, or
  address-sensitive, respectively.

- A struct type is illegal to copy, non-trivial to copy, or address-sensitive
  if it has a field whose type is illegal to copy, non-trivial to copy, or
  address-sensitive, respectively.

- A union type is both illegal and non-trivial to copy if it has a field whose
  type is non-trivial or illegal to copy.

- A union type is address-sensitive if it has a field whose type is
  address-sensitive.

- A program is ill-formed if it uses a type that is illegal to copy as
  a function parameter, argument, or return type.

- A program is ill-formed if an expression requires a type to be copied that is
  illegal to copy.

- Otherwise, copying a type that is non-trivial to copy correctly copies its
  subobjects.

- Types that are address-sensitive must always be passed and returned
  indirectly. Thus, changing the address-sensitivity of a type may be
  ABI-breaking even if its size and alignment do not change.

``<ptrauth.h>``
~~~~~~~~~~~~~~~

This header defines the following types and operations:

``ptrauth_key``
^^^^^^^^^^^^^^^

This ``enum`` is the type of abstract signing keys.  In addition to defining
the set of implementation-specific signing keys (for example, Armv8.3 defines
``ptrauth_key_asia``), it also defines some portable aliases for those keys.
For example, ``ptrauth_key_function_pointer`` is the key generally used for
C function pointers, which will generally be suitable for other
function-signing schemas.

In all the operation descriptions below, key values must be constant values
corresponding to one of the implementation-specific abstract signing keys from
this ``enum``.

``ptrauth_extra_data_t``
^^^^^^^^^^^^^^^^^^^^^^^^

This is a ``typedef`` of a standard integer type of the correct size to hold
a discriminator value.

In the signing and authentication operation descriptions below, discriminator
values must have either pointer type or integer type. If the discriminator is
an integer, it will be coerced to ``ptrauth_extra_data_t``.

``ptrauth_blend_discriminator``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c

  ptrauth_blend_discriminator(pointer, integer)

Produce a discriminator value which blends information from the given pointer
and the given integer.

Implementations may ignore some bits from each value, which is to say, the
blending algorithm may be chosen for speed and convenience over theoretical
strength as a hash-combining algorithm.  For example, arm64e simply overwrites
the high 16 bits of the pointer with the low 16 bits of the integer, which can
be done in a single instruction with an immediate integer.

``pointer`` must have pointer type, and ``integer`` must have integer type. The
result has type ``ptrauth_extra_data_t``.

``ptrauth_string_discriminator``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c

  ptrauth_string_discriminator(string)

Compute a constant discriminator from the given string.

``string`` must be a string literal of ``char`` character type.  The result has
type ``ptrauth_extra_data_t``.

The result value is never zero and always within range for both the
``__ptrauth`` qualifier and ``ptrauth_blend_discriminator``.

This can be used in constant expressions.

``ptrauth_strip``
^^^^^^^^^^^^^^^^^

.. code-block:: c

  ptrauth_strip(signedPointer, key)

Given that ``signedPointer`` matches the layout for signed pointers signed with
the given key, extract the raw pointer from it.  This operation does not trap
and cannot fail, even if the pointer is not validly signed.

``ptrauth_sign_constant``
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c

  ptrauth_sign_constant(pointer, key, discriminator)

Return a signed pointer for a constant address in a manner which guarantees
a non-attackable sequence.

``pointer`` must be a constant expression of pointer type which evaluates to
a non-null pointer.
``key``  must be a constant expression of type ``ptrauth_key``.
``discriminator`` must be a constant expression of pointer or integer type;
if an integer, it will be coerced to ``ptrauth_extra_data_t``.
The result will have the same type as ``pointer``.

This can be used in constant expressions.

``ptrauth_sign_unauthenticated``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c

  ptrauth_sign_unauthenticated(pointer, key, discriminator)

Produce a signed pointer for the given raw pointer without applying any
authentication or extra treatment.  This operation is not required to have the
same behavior on a null pointer that the language implementation would.

This is a treacherous operation that can easily result in `signing oracles`_.
Programs should use it seldom and carefully.

``ptrauth_auth_and_resign``
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c

  ptrauth_auth_and_resign(pointer, oldKey, oldDiscriminator, newKey, newDiscriminator)

Authenticate that ``pointer`` is signed with ``oldKey`` and
``oldDiscriminator`` and then resign the raw-pointer result of that
authentication with ``newKey`` and ``newDiscriminator``.

``pointer`` must have pointer type.  The result will have the same type as
``pointer``.  This operation is not required to have the same behavior on
a null pointer that the language implementation would.

The code sequence produced for this operation must not be directly attackable.
However, if the discriminator values are not constant integers, their
computations may still be attackable.  In the future, Clang should be enhanced
to guaranteed non-attackability if these expressions are
:ref:`safely-derived<Safe derivation>`.

``ptrauth_auth_function``
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c

  ptrauth_auth_function(pointer, key, discriminator)

Authenticate that ``pointer`` is signed with ``key`` and ``discriminator`` and
re-sign it to the standard schema for a function pointer of its type.

``pointer`` must have function pointer type.  The result will have the same
type as ``pointer``.  This operation is not required to have the same behavior
on a null pointer that the language implementation would.

This operation makes the same attackability guarantees as
``ptrauth_auth_and_resign``.

If this operation appears syntactically as the function operand of a call,
Clang guarantees that the call will directly authenticate the function value
using the given schema rather than re-signing to the standard schema.

``ptrauth_auth_data``
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c

  ptrauth_auth_data(pointer, key, discriminator)

Authenticate that ``pointer`` is signed with ``key`` and ``discriminator`` and
remove the signature.

``pointer`` must have object pointer type.  The result will have the same type
as ``pointer``.  This operation is not required to have the same behavior on
a null pointer that the language implementation would.

In the future when Clang makes safe derivation guarantees, the result of
this operation should be considered safely-derived.

``ptrauth_sign_generic_data``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c

  ptrauth_sign_generic_data(value1, value2)

Computes a signature for the given pair of values, incorporating a secret
signing key.

This operation can be used to verify that arbitrary data has not been tampered
with by computing a signature for the data, storing that signature, and then
repeating this process and verifying that it yields the same result.  This can
be reasonably done in any number of ways; for example, a library could compute
an ordinary checksum of the data and just sign the result in order to get the
tamper-resistance advantages of the secret signing key (since otherwise an
attacker could reliably overwrite both the data and the checksum).

``value1`` and ``value2`` must be either pointers or integers.  If the integers
are larger than ``uintptr_t`` then data not representable in ``uintptr_t`` may
be discarded.

The result will have type ``ptrauth_generic_signature_t``, which is an integer
type.  Implementations are not required to make all bits of the result equally
significant; in particular, some implementations are known to not leave
meaningful data in the low bits.

Standard ``__ptrauth`` qualifiers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``<ptrauth.h>`` additionally provides several macros which expand to
``__ptrauth`` qualifiers for common ABI situations.

For convenience, these macros expand to nothing when pointer authentication is
disabled.

These macros can be found in the header; some details of these macros may be
unstable or implementation-specific.


Theory of operation
-------------------

The threat model of pointer authentication is as follows:

- The attacker has the ability to read and write to a certain range of
  addresses, possibly the entire address space.  However, they are constrained
  by the normal rules of the process: for example, they cannot write to memory
  that is mapped read-only, and if they access unmapped memory it will trigger
  a trap.

- The attacker has no ability to add arbitrary executable code to the program.
  For example, the program does not include malicious code to begin with, and
  the attacker cannot alter existing instructions, load a malicious shared
  library, or remap writable pages as executable.  If the attacker wants to get
  the process to perform a specific sequence of actions, they must somehow
  subvert the normal control flow of the process.

In both of the above paragraphs, it is merely assumed that the attacker's
*current* capabilities are restricted; that is, their current exploit does not
directly give them the power to do these things.  The attacker's immediate goal
may well be to leverage their exploit to gain these capabilities, e.g. to load
a malicious dynamic library into the process, even though the process does not
directly contain code to do so.

Note that any bug that fits the above threat model can be immediately exploited
as a denial-of-service attack by simply performing an illegal access and
crashing the program.  Pointer authentication cannot protect against this.
While denial-of-service attacks are unfortunate, they are also unquestionably
the best possible result of a bug this severe. Therefore, pointer authentication
enthusiastically embraces the idea of halting the program on a pointer
authentication failure rather than continuing in a possibly-compromised state.

Pointer authentication is a form of control-flow integrity (CFI) enforcement.
The basic security hypothesis behind CFI enforcement is that many bugs can only
be usefully exploited (other than as a denial-of-service) by leveraging them to
subvert the control flow of the program.  If this is true, then by inhibiting or
limiting that subversion, it may be possible to largely mitigate the security
consequences of those bugs by rendering them impractical (or, ideally,
impossible) to exploit.

Every indirect branch in a program has a purpose.  Using human intelligence, a
programmer can describe where a particular branch *should* go according to this
purpose: a ``return`` in ``printf`` should return to the call site, a particular
call in ``qsort`` should call the comparator that was passed in as an argument,
and so on.  But for CFI to enforce that every branch in a program goes where it
*should* in this sense would require CFI to perfectly enforce every semantic
rule of the program's abstract machine; that is, it would require making the
programming environment perfectly sound.  That is out of scope.  Instead, the
goal of CFI is merely to catch attempts to make a branch go somewhere that its
obviously *shouldn't* for its purpose: for example, to stop a call from
branching into the middle of a function rather than its beginning.  As the
information available to CFI gets better about the purpose of the branch, CFI
can enforce tighter and tighter restrictions on where the branch is permitted to
go.  Still, ultimately CFI cannot make the program sound.  This may help explain
why pointer authentication makes some of the choices it does: for example, to
sign and authenticate mostly code pointers rather than every pointer in the
program.  Preventing attackers from redirecting branches is both particularly
important and particularly approachable as a goal.  Detecting corruption more
broadly is infeasible with these techniques, and the attempt would have far
higher cost.

Attacks on pointer authentication
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pointer authentication works as follows.  Every indirect branch in a program has
a purpose.  For every purpose, the implementation chooses a
:ref:`signing schema<Signing schemas>`.  At some place where a pointer is known
to be correct for its purpose, it is signed according to the purpose's schema.
At every place where the pointer is needed for its purpose, it is authenticated
according to the purpose's schema.  If that authentication fails, the program is
halted.

There are a variety of ways to attack this.

Attacks of interest to programmers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These attacks arise from weaknesses in the default protections offered by
pointer authentication.  They can be addressed by using attributes or intrinsics
to opt in to stronger protection.

Substitution attacks
++++++++++++++++++++

An attacker can simply overwrite a pointer intended for one purpose with a
pointer intended for another purpose if both purposes use the same signing
schema and that schema does not use address diversity.

The most common source of this weakness is when code relies on using the default
language rules for C function pointers.  The current implementation uses the
exact same signing schema for all C function pointers, even for functions of
substantially different type.  While efforts are ongoing to improve constant
diversity for C function pointers of different type, there are necessary limits
to this.  The C standard requires function pointers to be copyable with
``memcpy``, which means that function pointers can never use address diversity.
Furthermore, even if a function pointer can only be replaced with another
function of the exact same type, that can still be useful to an attacker, as in
the following example of a hand-rolled "v-table":

.. code-block:: c

  struct ObjectOperations {
    void (*retain)(Object *);
    void (*release)(Object *);
    void (*deallocate)(Object *);
    void (*logStatus)(Object *);
  };

The weakness in this design is that by lacking any context specific
discriminator, this means an attacker can substitute any of these fields with
any other function pointer signed with the default schema. Similarly the lack of
address diversity allows an attacker to replace the functions in one type's
"v-table" with those of another. This can be mitigated by overriding the default
authentication schema with a more specific signing schema for each purpose.  For
instance, in this example, the ``__ptrauth`` qualifier can be used with a
different constant discriminator for each field.  Since there's no particular
reason it's important for this v-table to be copyable with ``memcpy``, the
functions can also be signed with address diversity:

.. code-block:: c

  #if defined(__PTRAUTH__)
  #define objectOperation(discriminator) \
    __ptrauth(ptrauth_key_function_pointer, 1, discriminator)
  #else
  #define objectOperation(discriminator)
  #endif

  struct ObjectOperations {
    void (*objectOperation(0xf017) retain)(Object *);
    void (*objectOperation(0x2639) release)(Object *);
    void (*objectOperation(0x8bb0) deallocate)(Object *);
    void (*objectOperation(0xc5d4) logStatus)(Object *);
  };

This weakness can also sometimes be mitigated by simply keeping the signed
pointer in constant memory, but this is less effective than using better signing
diversity.

.. _Access path attacks:

Access path attacks
+++++++++++++++++++

If a signed pointer is often accessed indirectly (that is, by first loading the
address of the object where the signed pointer is stored), an attacker can
affect uses of it by overwriting the intermediate pointer in the access path.

The most common scenario exhibiting this weakness is an object with a pointer to
a "v-table" (a structure holding many function pointers). An attacker does not
need to replace a signed function pointer in the v-table if they can instead
simply replace the v-table pointer in the object with their own pointer ---
perhaps to memory where they've constructed their own v-table, or to existing
memory that coincidentally happens to contain a signed pointer at the right
offset that's been signed with the right signing schema.

This attack arises because data pointers are not signed by default. It works
even if the signed pointer uses address diversity: address diversity merely
means that each pointer is signed with its own storage address,
which (by design) is invariant to changes in the accessing pointer.

Using sufficiently diverse signing schemas within the v-table can provide
reasonably strong mitigation against this weakness.  Always use address and type
diversity in v-tables to prevent attackers from assembling their own v-table.
Avoid re-using constant discriminators to prevent attackers from replacing a
v-table pointer with a pointer to totally unrelated memory that just happens to
contain an similarly-signed pointer, or reused memory containing a different
type.

Further mitigation can be attained by signing pointers to v-tables. Any
signature at all should prevent attackers from forging v-table pointers; they
will need to somehow harvest an existing signed pointer from elsewhere in
memory.  Using a meaningful constant discriminator will force this to be
harvested from an object with similar structure (e.g. a different implementation
of the same interface).  Using address diversity will prevent such harvesting
entirely.  However, care must be taken when sourcing the v-table pointer
originally; do not blindly sign a pointer that is not
:ref:`safely derived<Safe derivation>`.

.. _Signing oracles:

Signing oracles
+++++++++++++++

A signing oracle is a bit of code which can be exploited by an attacker to sign
an arbitrary pointer in a way that can later be recovered.  Such oracles can be
used by attackers to forge signatures matching the oracle's signing schema,
which is likely to cause a total compromise of pointer authentication's
effectiveness.

This attack only affects ordinary programmers if they are using certain
treacherous patterns of code.  Currently this includes:

- all uses of the ``__ptrauth_sign_unauthenticated`` intrinsic and
- assigning values to ``__ptrauth``-qualified l-values.

Care must be taken in these situations to ensure that the pointer being signed
has been :ref:`safely derived<Safe derivation>` or is otherwise not possible to
attack.  (In some cases, this may be challenging without compiler support.)

A diagnostic will be added in the future for implicitly dangerous patterns of
code, such as assigning a non-safely-derived values to a
``__ptrauth``-qualified l-value.

.. _Authentication oracles:

Authentication oracles
++++++++++++++++++++++

An authentication oracle is a bit of code which can be exploited by an attacker
to leak whether a signed pointer is validly signed without halting the program
if it isn't.  Such oracles can be used to forge signatures matching the oracle's
signing schema if the attacker can repeatedly invoke the oracle for different
candidate signed pointers. This is likely to cause a total compromise of pointer
authentication's effectiveness.

There should be no way for an ordinary programmer to create an authentication
oracle using the current set of operations. However, implementation flaws in the
past have occasionally given rise to authentication oracles due to a failure to
immediately trap on authentication failure.

The likelihood of creating an authentication oracle is why there is currently no
intrinsic which queries whether a signed pointer is validly signed.


Attacks of interest to implementors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These attacks are not inherent to the model; they arise from mistakes in either
implementing or using the `sign` and `auth` operations. Avoiding these mistakes
requires careful work throughout the system.

Failure to trap on authentication failure
+++++++++++++++++++++++++++++++++++++++++

Any failure to halt the program on an authentication failure is likely to be
exploitable by attackers to create an
:ref:`authentication oracle<Authentication oracles>`.

There are several different ways to introduce this problem:

- The implementation might try to halt the program in some way that can be
  intercepted.

  For example, the Armv8.3 ``aut`` instructions do not directly trap on
  authentication failure on processors that lack the ``FPAC`` extension.
  Instead, they corrupt their results to be invalid pointers, with the idea that
  subsequent uses of those pointers will trigger traps as bad memory accesses.
  However, most kernels do not immediately halt programs that trap due to bad
  memory accesses; instead, they notify the process to give it an opportunity to
  recover. If this happens with an ``auth`` failure, the attacker may be able to
  exploit the recovery path in a way that creates an oracle. Kernels must
  provide a way for a process to trap unrecoverably, and this should cover all
  ``FPAC`` traps. Compilers must ensure that ``auth`` failures trigger an
  unrecoverable trap, ideally by taking advantage of ``FPAC``, but if necessary
  by emitting extra instructions.

- A compiler might use an intermediate representation (IR) for ``sign`` and
  ``auth`` operations that cannot make adequate correctness guarantees.

  For example, suppose that an IR uses ARMv8.3-like semantics for ``auth``: the
  operation merely corrupts its result on failure instead of promising to trap.
  A frontend might emit patterns of IR that always follow an ``auth`` with a
  memory access, thinking that this ensures correctness. But if the IR can be
  transformed to insert code between the ``auth`` and the access, or if the
  ``auth`` can be speculated, then this potentially creates an oracle.  It is
  better for ``auth`` to semantically guarantee to trap, potentially requiring
  an explicit check in the generated code. An ARMv8.3-like target can avoid this
  explicit check in the common case by recognizing the pattern of an ``auth``
  followed immediately by an access.

Attackable code sequences
+++++++++++++++++++++++++

If code that is part of a pointer authentication operation is interleaved with
code that may itself be vulnerable to attacks, an attacker may be able to use
this to create a :ref:`signing<Signing oracles>` or
:ref:`authentication<Authentication oracles>` oracle.

For example, suppose that the compiler is generating a call to a function and
passing two arguments: a signed constant pointer and a value derived from a
call.  In ARMv8.3, this code might look like so:

.. code-block:: asm

  adr x19, _callback.        ; compute &_callback
  paciza x19                 ; sign it with a constant discriminator of 0
  blr _argGenerator          ; call _argGenerator() (returns in x0)
  mov x1, x0                 ; move call result to second arg register
  mov x0, x19                ; move signed &_callback to first arg register
  blr _function              ; call _function

This code is correct, as would be a sequencing that does *both* the ``adr`` and
the ``paciza`` after the call to ``_argGenerator``.  But a sequence that
computes the address of ``_callback`` but leaves it as a raw pointer in a
register during the call to ``_argGenerator`` would be vulnerable:

.. code-block:: asm

  adr x19, _callback.        ; compute &_callback
  blr _argGenerator          ; call _argGenerator() (returns in x0)
  mov x1, x0                 ; move call result to second arg register
  paciza x19                 ; sign &_callback
  mov x0, x19                ; move signed &_callback to first arg register
  blr _function              ; call _function

If ``_argGenerator`` spills ``x19`` (a callee-save register), and if the
attacker can perform a write during this call, then the attacker can overwrite
the spill slot with an arbitrary pointer that will eventually be unconditionally
signed after the function returns.  This would be a signing oracle.

The implementation can avoid this by obeying two basic rules:

- The compiler's intermediate representations (IR) should not provide operations
  that expose intermediate raw pointers.  This may require providing extra
  operations that perform useful combinations of operations.

  For example, there should be an "atomic" auth-and-resign operation that should
  be used instead of emitting an ``auth`` operation whose result is fed into a
  ``sign``.

  Similarly, if a pointer should be authenticated as part of doing a memory
  access or a call, then the access or call should be decorated with enough
  information to perform the authentication; there should not be a separate
  ``auth`` whose result is used as the pointer operand for the access or call.
  (In LLVM IR, we do this for calls, but not yet for loads or stores.)

  "Operations" includes things like materializing a signed value to a known
  function or global variable.  The compiler must be able to recognize and emit
  this as a unified operation, rather than potentially splitting it up as in
  the example above.

- The compiler backend should not be too aggressive about scheduling
  instructions that are part of a pointer authentication operation. This may
  require custom code-generation of these operations in some cases.

Register clobbering
+++++++++++++++++++

As a refinement of the section on `Attackable code sequences`_, if the attacker
has the ability to modify arbitrary *register* state at arbitrary points in the
program, then special care must be taken.

For example, ARMv8.3 might materialize a signed function pointer like so:

.. code-block:: asm

  adr x0, _callback.        ; compute &_callback
  paciza x0                 ; sign it with a constant discriminator of 0

If an attacker has the ability to overwrite ``x0`` between these two
instructions, this code sequence is vulnerable to becoming a signing oracle.

For the most part, this sort of attack is not possible: it is a basic element of
the design of modern computation that register state is private and inviolable.
However, in systems that support asynchronous interrupts, this property requires
the cooperation of the interrupt-handling code. If that code saves register
state to memory, and that memory can be overwritten by an attacker, then
essentially the attack can overwrite arbitrary register state at an arbitrary
point.  This could be a concern if the threat model includes attacks on the
kernel or if the program uses user-space preemptive multitasking.

(Readers might object that an attacker cannot rely on asynchronous interrupts
triggering at an exact instruction boundary.  In fact, researchers have had some
success in doing exactly that.  Even ignoring that, though, we should aim to
protect against lucky attackers just as much as good ones.)

To protect against this, saved register state must be at least partially signed
(using something like `ptrauth_sign_generic_data`_).  This is required for
correctness anyway because saved thread states include security-critical
registers such as SP, FP, PC, and LR (where applicable).  Ideally, this
signature would cover all the registers, but since saving and restoring
registers can be very performance-sensitive, that may not be acceptable. It is
sufficient to set aside a small number of scratch registers that will be
guaranteed to be preserved correctly; the compiler can then be careful to only
store critical values like intermediate raw pointers in those registers.

``setjmp`` and ``longjmp`` should sign and authenticate the core registers (SP,
FP, PC, and LR), but they do not need to worry about intermediate values because
``setjmp`` can only be called synchronously, and the compiler should never
schedule pointer-authentication operations interleaved with arbitrary calls.

.. _Relative addresses:

Attacks on relative addressing
++++++++++++++++++++++++++++++

Relative addressing is a technique used to compress and reduce the load-time
cost of infrequently-used global data.  The pointer authentication system is
unlikely to support signing or authenticating a relative address, and in most
cases it would defeat the point to do so: it would take additional storage
space, and applying the signature would take extra work at load time.

Relative addressing is not precluded by the use of pointer authentication, but
it does take extra considerations to make it secure:

- Relative addresses must only be stored in read-only memory.  A writable
  relative address can be overwritten to point nearly anywhere, making it
  inherently insecure; this danger can only be compensated for with techniques
  for protecting arbitrary data like `ptrauth_sign_generic_data`_.

- Relative addresses must only be accessed through signed pointers with adequate
  diversity.  If an attacker can perform an `access path attack` to replace the
  pointer through which the relative address is accessed, they can easily cause
  the relative address to point wherever they want.

Signature forging
+++++++++++++++++

If an attacker can exactly reproduce the behavior of the signing algorithm, and
they know all the correct inputs to it, then they can perfectly forge a
signature on an arbitrary pointer.

There are three components to avoiding this mistake:

- The abstract signing algorithm should be good: it should not have glaring
  flaws which would allow attackers to predict its result with better than
  random accuracy without knowing all the inputs (like the key values).

- The key values should be kept secret.  If at all possible, they should never
  be stored in accessible memory, or perhaps only stored encrypted.

- Contexts that are meant to be independently protected should use different
  key values.  For example, the kernel should not use the same keys as user
  processes.  Different user processes should also use different keys from each
  other as much as possible, although this may pose its own technical
  challenges.

Remapping
+++++++++

If an attacker can change the memory protections on certain pages of the
program's memory, that can substantially weaken the protections afforded by
pointer authentication.

- If an attacker can inject their own executable code, they can also certainly
  inject code that can be used as a :ref:`signing oracle<Signing Oracles>`.
  The same is true if they can write to the instruction stream.

- If an attacker can remap read-only program data sections to be writable, then
  any use of :ref:`relative addresses` in global data becomes insecure.

- On platforms that use them, if an attacker can remap the memory containing
  the `global offset tables`_ as writable, then any unsigned pointers in those
  tables are insecure.

Remapping memory in this way often requires the attacker to have already
substantively subverted the control flow of the process.  Nonetheless, if the
operating system has a mechanism for mapping pages in a way that cannot be
remapped, this should be used wherever possible.

.. _Safe Derivation:

Safe derivation
~~~~~~~~~~~~~~~

Whether a data pointer is stored, even briefly, as a raw pointer can affect the
security-correctness of a program.  (Function pointers are never implicitly
stored as raw pointers; raw pointers to functions can only be produced with the
``<ptrauth.h>`` intrinsics.)  Repeated re-signing can also impact performance.
Clang makes a modest set of guarantees in this area:

- An expression of pointer type is said to be **safely derived** if:

  - it takes the address of a global variable or function, or

  - it is a load from a gl-value of ``__ptrauth``-qualified type, or

  - it is a load from read-only memory that has been initialized from a safely
    derived source, such as the `data const` section of a binary or library.

- If a value that is safely derived is assigned to a ``__ptrauth``-qualified
  object, including by initialization, then the value will be directly signed as
  appropriate for the target qualifier and will not be stored as a raw pointer.

- If the function expression of a call is a gl-value of ``__ptrauth``-qualified
  type, then the call will be authenticated directly according to the source
  qualifier and will not be resigned to the default rule for a function pointer
  of its type.

These guarantees are known to be inadequate for data pointer security. In
particular, Clang should be enhanced to make the following guarantees:

- A pointer should additionally be considered safely derived if it is:

  - the address of a gl-value that is safely derived,

  - the result of pointer arithmetic on a pointer that is safely derived (with
    some restrictions on the integer operand),

  - the result of a comma operator where the second operand is safely derived,

  - the result of a conditional operator where the selected operand is safely
    derived, or

  - the result of loading from a safely derived gl-value.

- A gl-value should be considered safely derived if it is:

  - a dereference of a safely derived pointer,

  - a member access into a safely derived gl-value, or

  - a reference to a variable.

- An access to a safely derived gl-value should be guaranteed to not allow
  replacement of any of the safely-derived component values at any point in the
  access.  "Access" should include loading a function pointer.

- Assignments should include pointer-arithmetic operators like ``+=``.

Making these guarantees will require further work, including significant new
support in LLVM IR.

Furthermore, Clang should implement a warning when assigning a data pointer that
is not safely derived to a ``__ptrauth``-qualified gl-value.


Language ABI
------------

This section describes the pointer-authentication ABI currently implemented in
Clang for the Apple arm64e target.  As other targets adopt pointer
authentication, this section should be generalized to express their ABIs as
well.

Key assignments
~~~~~~~~~~~~~~~

ARMv8.3 provides four abstract signing keys: ``IA``, ``IB``, ``DA``, and ``DB``.
The architecture designates ``IA`` and ``IB`` for signing code pointers and
``DA`` and ``DB`` for signing data pointers; this is reinforced by two
properties:

- The ISA provides instructions that perform combined auth+call and auth+load
  operations; these instructions can only use the ``I`` keys and ``D`` keys,
  respectively.

- AArch64's TBI feature can be separately enabled for code pointers (controlling
  whether indirect-branch instructions ignore those bits) and data pointers
  (controlling whether memory-access instructions) ignore those bits. If TBI is
  enabled for a kind of pointer, the sign and auth operations preserve the TBI
  bits when signing with an associated keys (at the cost of shrinking the number
  of available signing bits by 8).

arm64e then further subdivides the keys as follows:

- The ``A`` keys are used for primarily "global" purposes like signing v-tables
  and function pointers.  These keys are sometimes called *process-independent*
  or *cross-process* because on existing OSes they are not changed when changing
  processes, although this is not a platform guarantee.

- The ``B`` keys are used for primarily "local" purposes like signing return
  addresses.  These keys are sometimes called *process-specific* because they
  are typically different between processes. However, they are in fact shared
  across processes in one situation: systems which provide ``fork`` cannot
  change these keys in the child process; they can only be changed during
  ``exec``.

Implementation-defined algorithms and quantities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The cryptographic hash algorithm used to compute signatures in ARMv8.3 is a
private detail of the hardware implementation.

arm64e restricts constant discriminators (used in ``__ptrauth`` and
``ptrauth_blend_discriminator``) to the range from 0 to 65535, inclusive.  A 0
discriminator generally signifies that no blending is required; see the
documentation for ``ptrauth_blend_discriminator``.  This range is somewhat
narrow but has two advantages:

- The AArch64 ISA allows an arbitrary 16-bit immediate to be written over the
  top 16 bits of a register in a single instruction:

  .. code-block:: asm

    movk xN, #0x4849, LSL 48

  This is ideal for the discriminator blending operation because it adds minimal
  code-size overhead and avoids overwriting any interesting bits from the
  pointer.  Blending in a wider constant discriminator would either clobber
  interesting bits (e.g. if it was loaded with ``movk xN, #0x4c4f, LSL 32``) or
  require significantly more code (e.g. if the discriminator was loaded with a
  ``mov+bfi`` sequence).

- It is possible to pack a 16-bit discriminator into loader metadata with
  minimal compromises, whereas a wider discriminator would require extra
  metadata storage and therefore significantly impact load times.

The string hash used by ``ptrauth_string_discriminator`` is a 64-bit SipHash-2-4
using the constant seed ``b5d4c9eb79104a796fec8b1b428781d4`` (big-endian), with
the result reduced by modulo to the range of non-zero discriminators (i.e.
``(rawHash % 65535) + 1``).

Return addresses
~~~~~~~~~~~~~~~~

The kernel must ensure that attackers cannot replace LR due to an asynchronous
exception; see `Register clobbering`_.  If this is done by generally protecting
LR, then functions which don't spill LR to the stack can avoid signing it
entirely.  Otherwise, the return address must be signed; on arm64e it is signed
with the ``IB`` key using the stack pointer on entry as the discriminator.

Protecting return addresses is of such particular importance that the ``IB`` key
is almost entirely reserved for this purpose.

Global offset tables
~~~~~~~~~~~~~~~~~~~~

The global offset table (GOT) is not part of the language ABI, but it is a
common implementation technique for dynamic linking which deserves special
discussion here.

Whenever possible, signed pointers should be materialized directly in code
rather than via the GOT, e.g. using an ``adrp+add+pac`` sequence on ARMv8.3.
This decreases the amount of work necessary at load time to initialize the GOT,
but more importantly, it defines away the potential for several attacks:

- Attackers cannot change instructions, so there is no way to cause this code
  sequence to materialize a different pointer, whereas an access via the GOT
  always has *at minimum* a probabilistic chance to be the target of successful
  `substitution attacks`_.

- The GOT is a dense pool of fixed pointers at a fixed offset relative to code;
  attackers can search this pool for useful pointers that can be used in
  `substitution attacks`_, whereas pointers that are only materialized directly
  are not so easily available.

- Similarly, attackers can use `access path attacks`_ to replace a pointer to a
  signed pointer with a pointer to the GOT if the signing schema used within the
  GOT happens to be the same as the original pointer.  This kind of collision
  becomes much less likely to be useful the fewer pointers are in the GOT in the
  first place.

If this can be done for a symbol, then the compiler need only ensure that it
materializes the signed pointer using registers that are safe against
`register clobbering`_.

However, many symbols can only be accessed via the GOT, e.g. because they
resolve to definitions outside of the current image.  In this case, care must
be taken to ensure that using the GOT does not introduce weaknesses.

- If the entire GOT can be mapped read-only after loading, then no signing is
  required within the GOT.  In fact, not signing pointers in the GOT is
  preferable in this case because it makes the GOT useless for the harvesting
  and access-path attacks above.  Storing raw pointers in this way is usually
  extremely unsafe, but for the special case of an immutable GOT entry it's fine
  because the GOT is always accessed via an address that is directly
  materialized in code and thus provably unattackable.  (But see `Remapping`_.)

- Otherwise, GOT entries which are used for producing a signed pointer constant
  must be signed.  The signing schema used in the GOT need not match the target
  signing schema for the signed constant.  To counteract the threats of
  substitution attacks, it's best if GOT entries can be signed with address
  diversity.  Using a good constant discriminator as well (perhaps derived from
  the symbol name) can make it less useful to use a pointer to the GOT as the
  replacement in an :ref:`access path attack<Access path attacks>`.

In either case, the compiler must ensure that materializing the address of a GOT
entry as part of producing a signed pointer constant is not vulnerable to
`register clobbering`_.  If the linker also generates code for this, e.g. for
call stubs, this generated code must take the same precautions.

Dynamic symbol lookup
~~~~~~~~~~~~~~~~~~~~~

On platforms that support dynamically loading or resolving symbols it is
necessary for them to define the pointer authentication semantics of the APIs
provided to perform such lookups. While the platform may choose to reply
unsigned pointers from such function and rely on the caller performing the
initial signing, doing so creates the opportunity for caller side errors that
create :ref:`signing oracles<Signing Oracles>`.

On arm64e the `dlsym` function is used to resolve a symbol at runtime. If the
resolved symbol is a function or other code pointer the returned pointer is
signed using the default function signing schema described in
:ref:`C function pointers<C function abi>`. If the resolved symbol is not a code pointer it is
returned as an unsigned pointer.

.. _C function abi:

C function pointers
~~~~~~~~~~~~~~~~~~~

On arm64e, C function pointers are currently signed with the ``IA`` key without
address diversity and with a constant discriminator of 0.

The C and C++ standards do not permit C function pointers to be signed with
address diversity by default: in C++ terms, function pointer types are required
to be trivially copyable, which means they must be copyable with ``memcpy``.

The use of a uniform constant discriminator greatly simplifies the adoption of
arm64e, but it is a significant weakness in the mitigation because it allows any
C function pointer to be replaced with another. Clang supports
`-fptrauth-function-pointer-type-discrimination`, which enables a variant ABI
that uses type discrimination for function pointers. When generating the type
based discriminator for a function type all primitive integer types are
considered equivalent due to the prevalence of mismatching integer parameter
types in real world code. Type discrimination of function pointers is
ABI-incompatible with the standard arm64e ABI, but it can be used in constrained
contexts such as embedded systems or in code that does not require function
pointer interoperation with the standard ABI (e.g. because it does not pass
function pointers back and forth, or only does so through
``__ptrauth``-qualified l-values).

C++ virtual tables
~~~~~~~~~~~~~~~~~~

By default the pointer to a C++ virtual table is currently signed with the
``DA`` key, address diversity, and a constant discriminator equal to the string
hash (see `ptrauth_string_discriminator`_) of the mangled v-table identifier
of the primary base class for the v-table. To support existing code or ABI
constraints it is possible to use the `ptrauth_vtable_pointer` attribute to
override the schema used for the v-table pointer of the base type of
polymorphic class hierarchy. This attribute permits the configuration of the
key, address diversity mode, and any extra constant discriminator to be used.

Virtual functions in a C++ virtual table are signed with the ``IA`` key, address
diversity, and a constant discriminator equal to the string hash (see
`ptrauth_string_discriminator`_) of the mangled name of the function which
originally gave rise to the v-table slot.

C++ dynamic_cast
~~~~~~~~~~~~~~~~

C++'s ``dynamic_cast`` presents a difficulty relative to other polymorphic
languages that have a
`top type <https://en.wikipedia.org/wiki/Any_type>` as the use of declaration
diversity for v-table pointers results in distinct signing schemas for each
isolated type hierarchy. As a result it is not possible for the Itanium ABI
defined ``__dynamic_cast`` entry point to directly authenticate the v-table
pointer of the provided object.

The current implementation uses a forced authentication of the subject object's
v-table prior to invoking ``__dynamic_cast`` to partially verify that the
object's vtable is valid. The ``__dynamic_cast`` implementation currently relies
on this caller side check to limit the substitutability of the v-table pointer
with an incorrect or invalid v-table. The subsequent implementation of the
dynamic cast algorithm is built on pointer auth protected ``type_info`` objects.

In future a richer solution may be developed to support vending the correct
authentication schema directly to the ``dynamic_cast`` implementation.

C++ std::type_info v-table pointers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The v-table pointer of the ``std::type_info`` type is signed with the ``DA`` key
and no additional diversity.

C++ member function pointers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A member function pointer is signed with the ``IA`` key, no address diversity,
and a constant discriminator equal to the string hash
(see `ptrauth_string_discriminator`_) of the member pointer type.  Address
diversity is not permitted by C++ for member function pointers because they must
be trivially-copyable types.

The Itanium C++ ABI specifies that member function pointers to virtual functions
simply store an offset to the correct v-table slot.  This ABI cannot be used
securely with pointer authentication because there is no safe place to store the
constant discriminator for the target v-table slot: if it's stored with the
offset, an attacker can simply overwrite it with the right discriminator for the
offset.  Even if the programmer never uses pointers to virtual functions, the
existence of this code path makes all member function pointer dereferences
insecure.

arm64e changes this ABI so that virtual function pointers are stored using
dispatch thunks with vague linkage.  Because arm64e supports interoperation with
``arm64`` code when pointer authentication is disabled, an arm64e member
function pointer dereference still recognizes the virtual-function
representation but uses an bogus discriminator on that path that should always
trap if pointer authentication is enabled dynamically.

The use of dispatch thunks means that ``==`` on member function pointers is no
longer reliable for virtual functions, but this is acceptable because the
standard makes no guarantees about it in the first place.

The use of dispatch thunks also is required to support declaration specific
authentication schemas for v-table pointers.

C++ mangling
~~~~~~~~~~~~

When the ``__ptrauth`` qualifier appears in a C++ mangled name,
it is mangled as a vendor qualifier with the signature
``U9__ptrauthILj<key>ELb<addressDiscriminated>ELj<extraDiscriminator>EE``.

e.g. ``int * __ptrauth(1, 0, 1234)`` will be mangled as
``U9__ptrauthILj1ELb0ELj1234EE``.

If the vtable pointer authentication scheme of a polymorphic class is overridden
we mangle the override information with the vendor qualifier
``__vtptrauth(int key, bool addressDiscriminated, unsigned extraDiscriminator)``,
where the extra discriminator is the explicit value the specified discrimination
mode evalutes to.

Blocks
~~~~~~

Block pointers are data pointers which must interoperate with the ObjC `id` type
and therefore cannot be signed themselves. As blocks conform to the ObjC `id`
type, they contain an ``isa`` pointer signed as described
:ref:`below<Objc isa and super>`.

The invocation pointer in a block is signed with the ``IA`` key using address
diversity and a constant dicriminator of 0.  Using a uniform discriminator is
seen as a weakness to be potentially improved, but this is tricky due to the
subtype polymorphism directly permitted for blocks.

Block descriptors and ``__block`` variables can contain pointers to functions
that can be used to copy or destroy the object.  These functions are signed with
the ``IA`` key, address diversity, and a constant discriminator of 0.  The
structure of block descriptors is under consideration for improvement.

Objective-C runtime
~~~~~~~~~~~~~~~~~~~

In addition to the compile time ABI design, the Objective-C runtime provides
additional protection to methods and other metadata that have been loaded into
the Objective-C method cache; this protection is private to the runtime.

Objective-C methods
~~~~~~~~~~~~~~~~~~~

Objective-C method lists sign methods with the ``IA`` key using address
diversity and a constant discriminator of 0.  Using a uniform constant
discriminator is believed to be acceptable because these tables are only
accessed internally to the Objective-C runtime.

Objective-C class method list pointer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The method list pointer in Objective-C classes are signed with the ``DA`` key
using address diversity, and a constant discriminator of 0xC310.

Objective-C class read-only data pointer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The read-only data pointer in Objective-C classes are signed with the ``DA`` key
using address diversity, and a constant discriminator of 0x61F8.

.. _Objc isa and super:

Objective-C ``isa`` and ``super`` pointers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An Objective-C object's ``isa`` and ``super`` pointers are both signed with
the ``DA`` key using address diversity and constant discriminators of 0x6AE1
and 0x25DA respectively.

Objective-C ``SEL`` pointers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, the type of an Objective-C instance variable of type ``SEL``, when
the qualifiers do not include an explicit ``__ptrauth`` qualifier, is adjusted
to be qualified with ``__ptrauth(ptrauth_key_asdb, 1, 0x57C2)``.

This provides a measure of implicit at-rest protection to  Objective-C classes
that store selectors, as in the common target-action design pattern. This
prevents attackers from overriding the selector to invoke an arbitrary different
method, which is a major attack vector in Objective-C. Since ``SEL`` values are
not normally passed around as signed pointers, there is a
:ref:`signing oracle<Signing Oracles>` associated with the initialization of the
ivar, but the use of address and constant diversity limit the risks.

The implicit qualifier means that the type of the ivar does not match its
declaration, which can cause type errors if the address of the ivar is taken:

.. code-block:: ObjC

  @interface A : NSObject {
    SEL _s;
  }
  @end

  void f(SEL *);

  @implementation A
  -(void)g
  {
     f(&_s);
  }
  @end

To fix such an mismatch the schema macro from `<ptrauth.h>`:

.. code-block:: ObjC

  #include <ptrauth.h>

  void f(SEL __ptrauth_objc_sel*);

or less safely, and introducing the possibility of an
:ref:`signing or authentication oracle<Signing oracles>`, an unauthencaticated
temporary may be used as intermediate storage.

Alternative implementations
---------------------------

Signature storage
~~~~~~~~~~~~~~~~~

It is not critical for the security of pointer authentication that the
signature be stored "together" with the pointer, as it is in Armv8.3. An
implementation could just as well store the signature in a separate word, so
that the ``sizeof`` a signed pointer would be larger than the ``sizeof`` a raw
pointer.

Storing the signature in the high bits, as Armv8.3 does, has several trade-offs:

- Disadvantage: there are substantially fewer bits available for the signature,
  weakening the mitigation by making it much easier for an attacker to simply
  guess the correct signature.

- Disadvantage: future growth of the address space will necessarily further
  weaken the mitigation.

- Advantage: memory layouts don't change, so it's possible for
  pointer-authentication-enabled code (for example, in a system library) to
  efficiently interoperate with existing code, as long as pointer
  authentication can be disabled dynamically.

- Advantage: the size of a signed pointer doesn't grow, which might
  significantly increase memory requirements, code size, and register pressure.

- Advantage: the size of a signed pointer is the same as a raw pointer, so
  generic APIs which work in types like `void *` (such as `dlsym`) can still
  return signed pointers.  This means that clients of these APIs will not
  require insecure code in order to correctly receive a function pointer.

Hashing vs. encrypting pointers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Armv8.3 implements ``sign`` by computing a cryptographic hash and storing that
in the spare bits of the pointer.  This means that there are relatively few
possible values for the valid signed pointer, since the bits corresponding to
the raw pointer are known.  Together with an ``auth`` oracle, this can make it
computationally feasible to discover the correct signature with brute force.
(The implementation should of course endeavor not to introduce ``auth``
oracles, but this can be difficult, and attackers can be devious.)

If the implementation can instead *encrypt* the pointer during ``sign`` and
*decrypt* it during ``auth``, this brute-force attack becomes far less
feasible, even with an ``auth`` oracle.  However, there are several problems
with this idea:

- It's unclear whether this kind of encryption is even possible without
  increasing the storage size of a signed pointer.  If the storage size can be
  increased, brute-force attacks can be equally well mitigated by simply storing
  a larger signature.

- It would likely be impossible to implement a ``strip`` operation, which might
  make debuggers and other out-of-process tools far more difficult to write, as
  well as generally making primitive debugging more challenging.

- Implementations can benefit from being able to extract the raw pointer
  immediately from a signed pointer.  An Armv8.3 processor executing an
  ``auth``-and-load instruction can perform the load and ``auth`` in parallel;
  a processor which instead encrypted the pointer would be forced to perform
  these operations serially.
