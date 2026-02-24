======================
Lifetime Safety Analysis
======================

.. contents::
   :local:

Introduction
============

Clang Lifetime Safety Analysis is a C++ language extension which warns about
potential dangling pointer defects in code. The analysis aims to detect
when a pointer, reference or view type (such as ``std::string_view``) refers to an object
that is no longer alive, a condition that leads to use-after-free bugs and
security vulnerabilities. Common examples include pointers to stack variables
that have gone out of scope, fields holding views to stack-allocated objects
(dangling-field), returning pointers/references to stack variables 
(return stack address) or iterators into container elements invalidated by
container operations (e.g., ``std::vector::push_back``)

The analysis design is inspired by `Polonius, the Rust borrow checker <https://github.com/rust-lang/polonius>`_,
but adapted to C++ idioms and constraints, such as the lack of borrow checker exclusivity (alias-xor-mutability). 
Further details on the analysis method can be found in the `RFC on Discourse <https://discourse.llvm.org/t/rfc-intra-procedural-lifetime-analysis-in-clang/86291/>`_.

This is compile-time analysis; there is no run-time overhead. 
It tracks pointer validity through intra-procedural data-flow analysis, supporting a form of gradual typing. While it does
not require lifetime annotations to get started, in their absence, the analysis
treats function calls with opaque semantics, potentially missing dangling pointer issues or producing false positives. As more functions are annotated
with attributes like ``[[clang::lifetimebound]]``, ``[[gsl::Owner]]``, and
``[[gsl::Pointer]]``, the analysis can see through these contracts and enforce
lifetime safety at call sites with higher accuracy. This approach supports
gradual adoption in existing codebases. It is still very much under active development,
but it is mature enough to be used in production codebases.

Getting Started
----------------

.. code-block:: c++

  #include <string>
  #include <string_view>

  void simple_dangle() {
    std::string_view v;
    {
      std::string s = "hello";
      v = s;  // 'v' borrows from 's'.
    }       // 's' is destroyed here, 'v' becomes dangling.
    (void)v;  // WARNING! 'v' is used after 's' has been destroyed.
  }

This example demonstrates
a basic use-after-scope defect. The ``std::string_view`` object ``v`` holds a
reference to ``s``, a ``std::string``. When ``s`` goes out of
scope at the end of the inner block, ``v`` becomes a dangling reference, and
its subsequent use is flagged by the analysis.

Running The Analysis
--------------------

To run the analysis, compile with the ``-Wlifetime-safety`` flag, e.g.

.. code-block:: bash

  clang -c -Wlifetime-safety example.cpp

This flag enables a core set of lifetime safety checks. For more fine-grained
control over warnings, see :ref:`warning_flags`.

Lifetime Annotations
====================

While lifetime analysis can detect many issues without annotations, its
precision increases significantly when types and functions are annotated with
lifetime contracts. These annotations clarify ownership semantics and lifetime
dependencies, enabling the analysis to reason more accurately about pointer
validity across function calls.

Owner and Pointer Types
-----------------------

Lifetime analysis distinguishes between types that own the data they point to
(Owners) and types that are non-owning views or references to data owned by
others (Pointers). This distinction is made using GSL-style attributes:

*   ``[[gsl::Owner]]``: For types that manage the lifetime of a resource,
    like ``std::string``, ``std::vector``, ``std::unique_ptr``.
*   ``[[gsl::Pointer]]``: For non-owning types that borrow resources,
    like ``std::string_view``, ``gsl::span``, or raw pointers (which are
    implicitly treated as pointers).

Many common STL types, such as ``std::string_view`` and container iterators,
are automatically recognized as Pointers or Owners. You can annotate your own
types using these attributes:

.. code-block:: c++

  #include <string>
  #include <string_view>

  // Owning type
  struct [[gsl::Owner]] MyObj {
    std::string Data = "Hello";
  };

  // Non-owning view type
  struct [[gsl::Pointer]] View {
    std::string_view SV;
    View() = default;
    View(const MyObj& O) : SV(O.Data) {}
    void use() const {}
  };

  void test() {
    View v;
    {
      MyObj o;
      v = o;
    } // o is destroyed
    v.use(); // WARNING: object whose reference is captured does not live long enough
  }

Without these annotations, the analysis may not be able to determine whether a
type is owning or borrowing, which can affect analysis precision. For more
details on these attributes, see the Clang attribute reference for
`gsl::Owner <https://clang.llvm.org/docs/AttributeReference.html#gsl-owner>`_ and
`gsl::Pointer <https://clang.llvm.org/docs/AttributeReference.html#gsl-pointer>`_.

LifetimeBound
-------------

The ``[[clang::lifetimebound]]`` attribute can be applied to function parameters
or to the implicit ``this`` parameter of a method (by placing it after the
method declarator). It indicates that the returned pointer or reference is
valid only as long as the attributed parameter or ``this`` object is alive.
This is crucial for functions that return views or references to their
arguments.

.. code-block:: c++

  #include <string>
  #include <string_view>

  struct MyOwner {
    std::string s;
    std::string_view getView() const [[clang::lifetimebound]] { return s; }
  };

  void test_lifetimebound() {
    std::string_view sv;
    sv = MyOwner().getView(); // getView() is called on a temporary MyOwner
                             // MyOwner temporary is destroyed here.
    (void)sv;                // WARNING: object whose reference is captured does not live long enough
  }

Without ``[[clang::lifetimebound]]`` on ``getView()``, the analysis would not
know that the value returned by ``getView()`` depends on the temporary
``MyOwner`` object, and it would not be able to diagnose the dangling ``sv``.

For more details, see `lifetimebound <https://clang.llvm.org/docs/AttributeReference.html#lifetimebound>`_.

NoEscape
--------

The ``[[clang::noescape]]`` attribute can be applied to function parameters of
pointer or reference type. It indicates that the function will not allow the
parameter to escape its scope, for example, by returning it or assigning it to
a field or global variable. This is useful for parameters passed to callbacks
or visitors that are only used during the call and not stored.

For more details, see `noescape <https://clang.llvm.org/docs/AttributeReference.html#noescape>`_.

Checks Performed
================

Use-After-Scope
---------------

This is the simplest dangling pointer scenario, where a pointer or reference
outlives the stack variable it refers to.

.. code-block:: c++

  void use_after_scope() {
    int* p;
    {
      int i = 0;
      p = &i; // p borrows from i
    }       // i is destroyed, p dangles
    (void)*p; // WARNING: use-after-scope
  }

Return of stack address
-----------------------

This check warns when a function returns a pointer or reference to a
stack-allocated variable, which will be destroyed when the function returns,
leaving the caller with a dangling pointer.

.. code-block:: c++

  #include <string>
  #include <string_view>

  std::string_view return_stack_string_view() {
    std::string s = "hello";
    return s; // WARNING: address of stack memory is returned
  }

Dangling field
--------------

This check warns when a constructor or method assigns a pointer to a
stack-allocated variable or temporary to a field of the class, and the
stack variable's lifetime is shorter than the object's lifetime.

.. code-block:: c++

  #include <string>
  #include <string_view>

  struct DanglingField {
    std::string_view view;
    // WARNING: 's' is a temporary that will be destroyed after the
    // constructor finishes, leaving 'view' dangling.
    DanglingField(std::string s) : view(s) {}
  };

Use-after-invalidation (experimental)
-------------------------------------

This check warns when a reference to a container element (such as an iterator,
pointer or reference) is used after a container operation that may have
invalidated it. For example, adding elements to ``std::vector`` may cause
reallocation, invalidating all existing iterators, pointers and references to
its elements.

.. note::
  Container invalidation checking is highly experimental and may produce false
  positives or miss some invalidations. Field-sensitivity is also limited.

.. code-block:: c++

  #include <vector>

  void use_after_invalidation(std::vector<int>& v) {
    int* p = &v[0];
    v.push_back(4); // push_back might reallocate and invalidate p
    *p = 10;        // WARNING: use after invalidation
  }

Annotation Inference and Suggestions
====================================

In addition to detecting lifetime violations, the analysis can suggest adding
``[[clang::lifetimebound]]`` to function parameters or methods when it detects
that a pointer/reference to a parameter or ``this`` escapes via the return
value. This helps improve API contracts and allows the analysis to perform
more accurate checks in calling code.

To enable annotation suggestions, use ``-Wlifetime-safety-suggestions``.

.. code-block:: c++

  #include <string_view>

  // The analysis will suggest adding [[clang::lifetimebound]] to 'a'
  // because 'a' is returned.
  std::string_view return_view(std::string_view a) { // warning: parameter in intra-TU function should be marked [[clang::lifetimebound]]
    return a;               // note: param returned here
  }

TU-Wide analysis and Inference
------------------------------

By default, lifetime analysis is intra-procedural for error checking.
However, for annotation inference to be effective, lifetime information needs
to propagate across function calls. You can enable experimental
Translation-Unit-wide analysis using:

*   ``-flifetime-safety-inference``: Enables inference of ``lifetimebound``
    attributes across functions in a TU.
*   ``-fexperimental-lifetime-safety-tu-analysis``: Enables TU-wide analysis
    for better inference results.

.. _warning_flags:

Warning flags
=============

Lifetime safety warnings are organized into hierarchical groups, allowing users to
enable categories of checks incrementally. For example, ``-Wlifetime-safety``
enables all dangling pointer checks, while ``-Wlifetime-safety-permissive``
enables only the high-confidence subset of these checks.

*   **``-Wlifetime-safety-all``**: Enables all lifetime safety warnings, including
    dangling pointer checks, annotation suggestions, and annotation validations.

*   **``-Wlifetime-safety``**: Enables dangling pointer checks from both the
    ``permissive`` and ``strict`` groups listed below.
    *   **``-Wlifetime-safety-permissive``**: Enables high-confidence checks for dangling
        pointers. Recommended for initial adoption.
        *   **``-Wlifetime-safety-use-after-scope``**: Warns when a pointer to
            a stack variable is used after the variable's lifetime has ended.
        *   **``-Wlifetime-safety-return-stack-addr``**: Warns when a function
            returns a pointer or reference to one of its local stack variables.
        *   **``-Wlifetime-safety-dangling-field``**: Warns when a class field is
            assigned a pointer to a temporary or stack variable whose lifetime
            is shorter than the class instance.
    *   **``-Wlifetime-safety-strict``**: Enables stricter and experimental checks. These
        may produce false positives in code that uses move semantics heavily, as
        the analysis might conservatively assume a use-after-free even if
        ownership was transferred.
        *   **``-Wlifetime-safety-use-after-scope-moved``**: Same as
            ``-Wlifetime-safety-use-after-scope`` but for cases where the
            variable may have been moved from before its destruction.
        *   **``-Wlifetime-safety-return-stack-addr-moved``**: Same as
            ``-Wlifetime-safety-return-stack-addr`` but for cases where the
            variable may have been moved from.
        *   **``-Wlifetime-safety-dangling-field-moved``**: Same as
            ``-Wlifetime-safety-dangling-field`` but for cases where the
            variable may have been moved from.
        *   **``-Wlifetime-safety-invalidation``**: Warns when a container
            iterator or reference to an element is used after an operation
            that may invalidate it (Experimental).

*   **``-Wlifetime-safety-suggestions``**: Enables suggestions to add
    ``[[clang::lifetimebound]]`` to function parameters and ``this``
    parameters.
    *   **``-Wlifetime-safety-intra-tu-suggestions``**: Suggestions for functions
        local to the translation unit.
    *   **``-Wlifetime-safety-cross-tu-suggestions``**: Suggestions for functions
        visible across translation units (e.g., in headers).

*   **``-Wlifetime-safety-validations``**: Enables checks that validate existing
    lifetime annotations.
    *   **``-Wlifetime-safety-noescape``**: Warns when a parameter marked with
        ``[[clang::noescape]]`` escapes the function.

Limitations
===========

Move Semantics and False Positives
----------------------------------
When an object is moved from, its state becomes unspecified. If pointers or
views were created that refer to the object *before* it was moved, those
pointers may become invalid after the move. Because the analysis cannot always
know if a move operation invalidates outstanding pointers or simply transfers
ownership, it issues ``-Wlifetime-safety-*-moved`` warnings in these situations.
These warnings indicate a *potential* dangling issue but may be false positives
if ownership was safely transferred and the resource remains alive.
``std::unique_ptr::release()`` is treated similarly to ``std::move()`` in this
regard, as it also relinquishes ownership.

To avoid these warnings and prevent potential bugs, follow the
**"move-first-then-alias"** pattern: ensure that views or raw pointers are
created *after* a potential move, sourcing them from the new owner rather than
aliasing an object that is about to be moved.

For example, when initializing fields in a constructor, moving from a parameter *after* using it to initialize a view field can lead to warnings:

.. code-block:: c++

  #include <string>
  #include <string_view>
  #include <utility>

  struct BadFieldOrder {
    std::string_view view;
    std::string s_owned;
    // WARNING: 'view' is initialized from 's', then 's' is moved-from,
    // leaving 'view' pointing to a moved-from string.
    BadFieldOrder(std::string s) : view(s), s_owned(std::move(s)) {} // -Wlifetime-safety-dangling-field-moved
  };

  // CORRECT: Move 's' into 's_owned' first, then initialize 'view' from 's_owned'.
  struct GoodFieldOrder {
    std::string s_owned;
    std::string_view view;
    GoodFieldOrder(std::string s) : s_owned(std::move(s)), view(s_owned) {} // OK
  };

The same principle applies when creating other aliases via ``get()`` or ``release()`` before moving or releasing ownership:

.. code-block:: c++

  #include <memory>
  #include <utility>

  void use(int*);
  void take_ownership(std::unique_ptr<int>);

  void test_aliasing_before_move() {
    int* p;
    {
      auto u = std::make_unique<int>(1);
      p = u.get(); // p aliases u's content
      take_ownership(std::move(u)); // u is moved-from
    }
    // 'p' now points to memory whose ownership was transferred,
    // and it might be invalid depending on what take_ownership does.
    use(p); // WARNING: -Wlifetime-safety-use-after-scope-moved
  }

Dangling Fields and Intra-Procedural Analysis
---------------------------------------------
The lifetime analysis is intra-procedural. It analyzes one function or method at
a time.
This means if a field is assigned a pointer to a local variable or temporary
inside a constructor or method, and that local's lifetime ends before the method
returns, the analysis will issue a ``-Wlifetime-safety-dangling-field`` warning.
It must do so even if no *other* method of the class ever accesses this field,
because it cannot see how other methods are implemented or used.

.. code-block:: c++

  #include <string>
  #include <string_view>

  struct MyWidget {
    std::string_view name_;
    MyWidget(std::string name) : name_(name) {} // WARNING: 'name' is destroyed when ctor ends, leaving 'name_' dangling
    const char* data() { return name_.data(); } // Potential use-after-free if called
  };

In this case, ``name_`` dangles after the constructor finishes.
Even if ``data()`` is never called, the analysis flags the dangling assignment
in the constructor because it represents a latent bug.
The recommended approach is to ensure fields only point to objects that outlive
the field itself, for example by storing an owned object (e.g., ``std::string``)
or ensuring the borrowed object (e.g., one passed by ``const&``) has a
sufficient lifetime.


Heap and Globals
----------------

Currently, the analysis focuses on dangling pointers to stack variables,
temporaries, and function parameters. It does not track lifetimes of heap-
allocated memory or global variables.

Performance
===========

Lifetime analysis relies on Clang's CFG (Control Flow Graph). For functions
with very large or complex CFGs, analysis time can be significant. To mitigate
this, the analysis will skip functions where the number of CFG blocks exceeds
a certain threshold, controlled by the ``-flifetime-safety-max-cfg-blocks=N`` language
option.
