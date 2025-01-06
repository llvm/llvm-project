========================
Function Effect Analysis
========================

.. contents::
  :depth: 3
  :local:


Introduction
============

Clang Function Effect Analysis is a language extension which can warn about "unsafe"
constructs. The feature is currently tailored for the Performance Constraint attributes
``nonblocking`` and ``nonallocating``; functions with these attributes are verified as not
containing any language constructs or calls to other functions which violate the constraint.
(See :doc:`AttributeReference`.)


The ``nonblocking`` and ``nonallocating`` attributes
====================================================

Attribute syntax
----------------

The ``nonblocking`` and ``nonallocating`` attributes apply to function types, allowing them to be
attached to functions, blocks, function pointers, lambdas, and member functions.

.. code-block:: c++

  // Functions
  void nonblockingFunction() [[clang::nonblocking]];
  void nonallocatingFunction() [[clang::nonallocating]];

  // Function pointers
  void (*nonblockingFunctionPtr)() [[clang::nonblocking]];

  // Typedefs, type aliases.
  typedef void (*NBFunctionPtrTypedef)() [[clang::nonblocking]];
  using NBFunctionPtrTypeAlias_gnu = __attribute__((nonblocking)) void (*)();
  using NBFunctionPtrTypeAlias_std = void (*)() [[clang::nonblocking]];

  // C++ methods
  struct Struct {
    void NBMethod() [[clang::nonblocking]];
  };

  // C++ lambdas
  auto nbLambda = []() [[clang::nonblocking]] {};

  // Blocks
  void (^nbBlock)() = ^() [[clang::nonblocking]] {};

The attribute applies only to the function itself. In particular, it does not apply to any nested
functions or declarations, such as blocks, lambdas, and local classes.

This document uses the C++/C23 syntax ``[[clang::nonblocking]]``, since it parallels the placement
of the ``noexcept`` specifier, and the attributes have other similarities to ``noexcept``. The GNU
``__attribute__((nonblocking))`` syntax is also supported. Note that it requires a different
placement on a C++ type alias.

Like ``noexcept``, ``nonblocking`` and ``nonallocating`` have an optional argument, a compile-time
constant boolean expression. By default, the argument is ``true``, so ``[[clang::nonblocking]]``
is equivalent to ``[[clang::nonblocking(true)]]``, and declares the function type as never blocking.


Attribute semantics
-------------------

Together with ``noexcept``, the ``nonallocating`` and ``nonblocking`` attributes define an ordered
series of performance constraints. From weakest to strongest:

- ``noexcept`` (as per the C++ standard): The function type will never throw an exception.
- ``nonallocating``: The function type will never allocate memory on the heap or throw an
  exception.
- ``nonblocking``: The function type will never block on a lock, allocate memory on the heap,
  or throw an exception.

``nonblocking`` includes the ``nonallocating`` guarantee.

While ``nonblocking`` and ``nonallocating`` are conceptually a superset of ``noexcept``, neither
attribute implicitly specifies ``noexcept``. Further, ``noexcept`` has a specified runtime behavior of
aborting if an exception is thrown, while the ``nonallocating`` and ``nonblocking`` attributes are
mainly for compile-time analysis and have no runtime behavior, except in code built
with Clang's :doc:`RealtimeSanitizer`. Nonetheless, Clang emits a
warning if, in C++, a function is declared ``nonblocking`` or ``nonallocating`` without
``noexcept``. This diagnostic is controlled by ``-Wperf-constraint-implies-noexcept``.

``nonblocking(true)`` and ``nonallocating(true)`` apply to function *types*, and by extension, to
function-like declarations. When applied to a declaration with a body, the compiler verifies the
function, as described in the section "Analysis and warnings", below.

``blocking`` and ``allocating`` are synonyms for ``nonblocking(false)`` and
``nonallocating(false)``, respectively. They can be used on a function-like declaration to
explicitly disable any potential inference of ``nonblocking`` or ``nonallocating`` during
verification. (Inference is described later in this document). ``nonblocking(false)`` and
``nonallocating(false)`` are legal, but superfluous  when applied to a function *type*
that is not part of a declarator: ``float (int) [[nonblocking(false)]]`` and
``float (int)`` are identical types.

For functions with no explicit performance constraint, the worst is assumed: the function
allocates memory and potentially blocks, unless it can be inferred otherwise. This is detailed in the
discussion of verification.

The following example describes the meanings of all permutations of the two attributes and arguments:

.. code-block:: c++

  void nb1_na1() [[clang::nonblocking(true)]] [[clang::nonallocating(true)]];
  // Valid; nonallocating(true) is superfluous but doesn't contradict the guarantee.

  void nb1_na0() [[clang::nonblocking(true)]] [[clang::nonallocating(false)]];
  // error: 'allocating' and 'nonblocking' attributes are not compatible

  void nb0_na1() [[clang::nonblocking(false)]] [[clang::nonallocating(true)]];
  // Valid; the function does not allocate memory, but may lock for other reasons.

  void nb0_na0() [[clang::nonblocking(false)]] [[clang::nonallocating(false)]];
  // Valid.


Type conversions
----------------

A performance constraint can be removed or weakened via an implicit conversion. An attempt to add
or strengthen a performance constraint is unsafe and results in a warning. The rules for this
are comparable to that for ``noexcept`` in C++17 and later.

.. code-block:: c++

  void unannotated();
  void nonblocking() [[clang::nonblocking]];
  void nonallocating() [[clang::nonallocating]];

  void example()
  {
    // It's fine to remove a performance constraint.
    void (*fp_plain)();
    fp_plain = unannotated;
    fp_plain = nonblocking;
    fp_plain = nonallocating;

    // Adding/spoofing nonblocking is unsafe.
    void (*fp_nonblocking)() [[clang::nonblocking]];
    fp_nonblocking = nullptr;
    fp_nonblocking = nonblocking;
    fp_nonblocking = unannotated;
    // ^ warning: attribute 'nonblocking' should not be added via type conversion
    fp_nonblocking = nonallocating;
    // ^ warning: attribute 'nonblocking' should not be added via type conversion

    // Adding/spoofing nonallocating is unsafe.
    void (*fp_nonallocating)() [[clang::nonallocating]];
    fp_nonallocating = nullptr;
    fp_nonallocating = nonallocating;
    fp_nonallocating = nonblocking; // no warning because nonblocking includes nonallocating
    fp_nonallocating = unannotated;
    // ^ warning: attribute 'nonallocating' should not be added via type conversion
  }

Virtual methods
---------------

In C++, when a virtual method has a performance constraint, overriding methods in
subclasses inherit the constraint.

.. code-block:: c++

  struct Base {
    virtual void unsafe();
    virtual void safe() noexcept [[clang::nonblocking]];
  };

  struct Derived : public Base {
    void unsafe() [[clang::nonblocking]] override;
    // It's okay for an overridden method to be more constrained

    void safe() noexcept override;
    // This method is implicitly declared `nonblocking`, inherited from Base.
  };

Redeclarations, overloads, and name mangling
--------------------------------------------

The ``nonblocking`` and ``nonallocating`` attributes, like ``noexcept``, do not factor into
argument-dependent lookup and overloaded functions/methods.

First, consider that ``noexcept`` is integral to a function's type:

.. code-block:: c++

  void f1(int);
  void f1(int) noexcept;
  // error: exception specification in declaration does not match previous
  //   declaration

Unlike ``noexcept``, a redeclaration of ``f2`` with an added or stronger performance constraint is
legal and propagates the attribute to the previous declaration:

.. code-block:: c++

  int f2();
  int f2() [[clang::nonblocking]]; // redeclaration with stronger constraint is OK.

This greatly eases adoption by making it possible to annotate functions in external libraries
without modifying library headers.

A redeclaration with a removed or weaker performance constraint produces a warning, paralleling
the behavior of ``noexcept``:

.. code-block:: c++

  int f2() { return 42; }
  // warning: attribute 'nonblocking' on function does not match previous declaration

In C++14, the following two declarations of `f3` are identical (a single function). In C++17 they
are separate overloads:

.. code-block:: c++

  void f3(void (*)());
  void f3(void (*)() noexcept);

Similarly, the following two declarations of `f4` are separate overloads. This pattern may pose
difficulties due to ambiguity:

.. code-block:: c++

  void f4(void (*)());
  void f4(void (*)() [[clang::nonblocking]]);

The attributes have no effect on the mangling of function and method names.

Objective-C
-----------

The attributes are currently unsupported on Objective-C methods.

Analysis and warnings
=====================

Constraints
-----------

Functions declared ``nonallocating`` or ``nonblocking``, when defined, are verified according to the
following rules. Such functions:

1. May not allocate or deallocate memory on the heap. The analysis follows the calls to
   ``operator new`` and ``operator delete`` generated by the ``new`` and ``delete`` keywords, and
   treats them like any other function call. The global ``operator new`` and ``operator delete``
   aren't declared ``nonblocking`` or ``nonallocating`` and so they are considered unsafe. (This
   is correct because most memory allocators are not lock-free. Note that the placement form of
   ``operator new`` is implemented inline in libc++'s ``<new>`` header, and is verifiably
   ``nonblocking``, since it merely casts the supplied pointer to the result type.)

2. May not throw or catch exceptions. To throw, the compiler must allocate the exception on the
   heap. (Also, many subclasses of ``std::exception`` allocate a string). Exceptions are
   deallocated when caught.

3. May not make any indirect function call, via a virtual method, function pointer, or
   pointer-to-member function, unless the target is explicitly declared with the same
   ``nonblocking`` or ``nonallocating`` attribute (or stronger).

4. May not make direct calls to any other function, with the following exceptions:

  a. The callee is also explicitly declared with the same ``nonblocking`` or ``nonallocating``
     attribute (or stronger).
  b. The callee is defined in the same translation unit as the caller, does not have the ``false``
     form of the required attribute, and can be verified to have the same attribute or stronger,
     according to these same rules.
  c. The callee is a built-in function that is known not to block or allocate.
  d. The callee is declared ``noreturn`` and, if compiling C++, the callee is also declared
     ``noexcept``. This special case excludes functions such as ``abort()`` and ``std::terminate()``
     from the analysis. (The reason for requiring ``noexcept`` in C++ is that a function declared
     ``noreturn`` could be a wrapper for ``throw``.)

5. May not invoke or access an Objective-C method or property, since ``objc_msgSend()`` calls into
   the Objective-C runtime, which may allocate memory or otherwise block.

6. May not access thread-local variables. Typically, thread-local variables are allocated on the
   heap when first accessed.

Functions declared ``nonblocking`` have an additional constraint:

7. May not declare static local variables (e.g. Meyers singletons). The compiler generates a lock
   protecting the initialization of the variable.

Violations of any of these rules result in warnings, in the ``-Wfunction-effects`` category:

.. code-block:: c++

  void notInline();

  void example() [[clang::nonblocking]]
  {
    auto* x = new int;
    // warning: function with 'nonblocking' attribute must not allocate or deallocate
    //   memory

    if (x == nullptr) {
      static Logger* logger = createLogger();
      // warning: function with 'nonblocking' attribute must not have static local variables

      throw std::runtime_warning{ "null" };
      // warning: 'nonblocking" function 'example' must not throw exceptions
    }
    notInline();
    // warning: 'function with 'nonblocking' attribute must not call non-'nonblocking' function
    //   'notInline'
    // note (on notInline()): declaration cannot be inferred 'nonblocking' because it has no
    //   definition in this translation unit
  }

Inferring ``nonblocking`` or ``nonallocating``
----------------------------------------------

In the absence of a ``nonblocking`` or ``nonallocating`` attribute (whether ``true`` or ``false``),
a function that is called from a performance-constrained function may be analyzed to
infer whether it has a desired attribute. This analysis happens when the function is not a virtual
method, and it has a visible definition within the current translation unit (i.e. its body can be
traversed).

.. code-block:: c++

  void notInline();
  int implicitlySafe() { return 42; }
  void implicitlyUnsafe() { notInline(); }

  void example() [[clang::nonblocking]]
  {
    int x = implicitlySafe(); // OK
    implicitlyUnsafe();
    // warning: function with 'nonblocking' attribute must not call non-'nonblocking' function
    //   'implicitlyUnsafe'
    // note (on implicitlyUnsafe): function cannot be inferred 'nonblocking' because it calls
    //   non-'nonblocking' function 'notInline'
    // note (on notInline()): declaration cannot be inferred 'nonblocking' because it has no
    //   definition in this translation unit
  }

Lambdas and blocks
------------------

As mentioned earlier, the performance constraint attributes apply only to a single function and not
to any code nested inside it, including blocks, lambdas, and local classes. It is possible for a
nonblocking function to schedule the execution of a blocking lambda on another thread. Similarly, a
blocking function may create a ``nonblocking`` lambda for use in a realtime context.

Operations which create, destroy, copy, and move lambdas and blocks are analyzed in terms of the
underlying function calls. For example, the creation of a lambda with captures generates a function
call to an anonymous struct's constructor, passing the captures as parameters.

Implicit function calls in the AST
----------------------------------

The ``nonblocking`` / ``nonallocating`` analysis occurs at the Sema phase of analysis in Clang.
During Sema, there are some constructs which will eventually become function calls, but do not
appear as function calls in the AST. For example, ``auto* foo = new Foo;`` becomes a declaration
containing a ``CXXNewExpr`` which is understood as a function call to the global ``operator new``
(in this example), and a ``CXXConstructExpr``, which, for analysis purposes, is a function call to
``Foo``'s constructor. Most gaps in the analysis would be due to incomplete knowledge of AST
constructs which become function calls.

Disabling diagnostics
---------------------

Function effect diagnostics are controlled by ``-Wfunction-effects``.

A construct like this can be used to exempt code from the checks described here:

.. code-block:: c++

  #define NONBLOCKING_UNSAFE(...)                                    \
    _Pragma("clang diagnostic push")                                 \
    _Pragma("clang diagnostic ignored \"-Wunknown-warning-option\"") \
    _Pragma("clang diagnostic ignored \"-Wfunction-effects\"")       \
    __VA_ARGS__                                                      \
    _Pragma("clang diagnostic pop")

Disabling the diagnostic allows for:

- constructs which do block, but which in practice are used in ways to avoid unbounded blocking,
  e.g. a thread pool with semaphores to coordinate multiple realtime threads;
- using libraries which are safe but not yet annotated;
- incremental adoption in a large codebase.

Adoption
========

There are a few common issues that arise when adopting the ``nonblocking`` and ``nonallocating``
attributes.

C++ exceptions
--------------

Exceptions pose a challenge to the adoption of the performance constraints. Common library functions
which throw exceptions include:

+----------------------------------+-----------------------------------------------------------------------+
| Method                           | Alternative                                                           |
+==================================+=======================================================================+
| ``std::vector<T>::at()``         | ``operator[](size_t)``, after verifying that the index is in range.   |
+----------------------------------+-----------------------------------------------------------------------+
| ``std::optional<T>::value()``    | ``operator*``, after checking ``has_value()`` or ``operator bool()``. |
+----------------------------------+-----------------------------------------------------------------------+
| ``std::expected<T, E>::value()`` | Same as for ``std::optional<T>::value()``.                            |
+----------------------------------+-----------------------------------------------------------------------+


``std::function<R(Args...)>``
-----------------------------

``std::function<R(Args...)>`` is generally incompatible with ``nonblocking`` and ``nonallocating``
code, because a typical implementation may allocate heap memory in the constructor.

Alternatives:

- ``std::function_ref`` (available in C++26 or as ``llvm::function_ref``). This is appropriate and
  optimal when a functor's lifetime does not need to extend past the function that created it.

- ``inplace_function`` from WG14. This solves the allocation problem by giving the functor wrapper
  a fixed size known at compile time and using an inline buffer.

While these alternatives both address the heap allocation of ``std::function``, they are still
obstacles to ``nonblocking/nonallocating`` verification, for reasons detailed in the next section.


Interactions with type-erasure techniques
-----------------------------------------

``std::function<R(Args...)>`` illustrates a common C++ type-erasure technique. Using template
argument deduction, it decomposes a function type into its return and parameter types. Additional
components of the function type, including ``noexcept``, ``nonblocking``, ``nonallocating``, and any
other attributes, are discarded.

Standard library support for these components of a function type is not immediately forthcoming.

Code can work around this limitation in either of two ways:

1. Avoid abstractions like ``std::function`` and instead work directly with the original lambda type.

2. Create a specialized alternative, e.g. ``nonblocking_function_ref<R(Args...)>`` where all function
   pointers used in the implementation and its interface are ``nonblocking``.

As an example of the first approach, when using a lambda as a *Callable* template parameter, the
attribute is preserved:

.. code-block:: c++

  std::sort(vec.begin(), vec.end(),
    [](const Elem& a, const Elem& b) [[clang::nonblocking]] { return a.mem < b.mem; });

Here, the type of the ``Compare`` template parameter is an anonymous class generated from the
lambda, with an ``operator()`` method holding the ``nonblocking`` attribute.

A complication arises when a *Callable* template parameter, instead of being a lambda or class
implementing ``operator()``, is a function pointer:

.. code-block:: c++

  static bool compare_elems(const Elem& a, const Elem& b) [[clang::nonblocking]] {
    return a.mem < b.mem; };

  std::sort(vec.begin(), vec.end(), compare_elems);

Here, the type of ``compare_elems`` is decomposed to ``bool(const Elem&, const Elem&)``, without
``nonblocking``, when forming the template parameter. This can be solved using the second approach,
creating a specialized alternative which explicitly requires the attribute. In this case, it's
possible to use a small wrapper to transform the function pointer into a functor:

.. code-block:: c++

  template <typename>
  class nonblocking_fp;

  template <typename R, typename... Args>
  class nonblocking_fp<R(Args...)> {
  public:
    using impl_t = R (*)(Args...) [[clang::nonblocking]];

  private:
    impl_t mImpl{ nullptr_t };
  public:
    nonblocking_fp() = default;
    nonblocking_fp(impl_t f) : mImpl{ f } {}

    R operator()(Args... args) const
    {
      return mImpl(std::forward<Args>(args)...);
    }
  };

  // deduction guide (like std::function's)
  template< class R, class... ArgTypes >
  nonblocking_fp( R(*)(ArgTypes...) ) -> nonblocking_fp<R(ArgTypes...)>;

  // --

  // Wrap the function pointer in a functor which preserves ``nonblocking``.
  std::sort(vec.begin(), vec.end(), nonblocking_fp{ compare_elems });

Now, the ``nonblocking`` attribute of ``compare_elems`` is verified when it is converted to a
``nonblocking`` function pointer, as the argument to ``nonblocking_fp``'s constructor. The template
parameter is the functor class ``nonblocking_fp``.


Static local variables
----------------------

Static local variables are often used for lazily-constructed globals (Meyers singletons). Beyond the
compiler's use of a lock to ensure thread-safe initialization, it is dangerously easy to
inadvertently trigger initialization, involving heap allocation, from a ``nonblocking`` or
``nonallocating`` context.

Generally, such singletons need to be replaced by globals, and care must be taken to ensure their
initialization before they are used from ``nonblocking`` or ``nonallocating`` contexts.


Annotating libraries
--------------------

It can be surprising that the analysis does not depend on knowledge of any primitives; it simply
assumes the worst, that all function calls are unsafe unless explicitly marked as safe or able to be
inferred as safe. With ``nonblocking``, this appears to suffice for all but the most primitive of
spinlocks.

At least for an operating system's C functions, it is possible to define an override header which
redeclares safe common functions (e.g. ``pthread_self()``) with the addition of ``nonblocking``.
This may help in adopting the feature incrementally.

It also helps that many of the functions in the standard C libraries (notably ``<math.h>``)
are treated as built-in functions by Clang, which the diagnosis understands to be safe.

Much of the C++ standard library consists of inline templated functions which work well with
inference. A small number of primitives may need explicit ``nonblocking/nonallocating`` attributes.
