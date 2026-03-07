========================
Lifetime Safety Analysis
========================

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
      v = s;  // warning: object whose reference is captured does not live long enough
    }         // note: destroyed here
    std::cout << v; // note: later used here
  }

This example demonstrates
a basic use-after-scope bug. The ``std::string_view`` object ``v`` holds a
reference to ``s``, a ``std::string``. When ``s`` goes out of
scope at the end of the inner block, ``v`` becomes a dangling reference.
The analysis flags the assignment ``v = s`` as defective because ``s`` is
destroyed while ``v`` is still alive and points to ``s``, and adds a note
to where ``v`` is used after ``s`` has been destroyed.

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

  // Owner type
  struct [[gsl::Owner]] MyObj {
    std::string Data = "Hello";
  };

  // View type
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
      v = o; // warning: object whose reference is captured does not live long enough
    }        // note: destroyed here
    v.use(); // note: later used here
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
                             // warning: object whose reference is captured does not live long enough
                             // note: destroyed here
    (void)sv;                // note: later used here
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


.. raw:: html

   <style>
   /* Align text to left and add red/green colors */
   table.colored-code-table td, table.colored-code-table th { text-align: left !important; }
   table.colored-code-table td:first-child, table.colored-code-table th:first-child { background-color: #ffeaea !important; }
   table.colored-code-table td:nth-child(2), table.colored-code-table th:nth-child(2) { background-color: #eafaea !important; }
   table.colored-code-table td .highlight, table.colored-code-table td pre { background-color: transparent !important; border: none !important; }

   div.bad-code { background-color: #ffeaea !important; padding: 5px; border-left: 4px solid #ff6b6b; text-align: left !important; }
   div.bad-code .highlight, div.bad-code pre { background-color: transparent !important; border: none !important; }

   div.good-code { background-color: #eafaea !important; padding: 5px; border-left: 4px solid #51cf66; text-align: left !important; }
   div.good-code .highlight, div.good-code pre { background-color: transparent !important; border: none !important; }
   </style>

Use after scope
---------------

This is the simplest dangling pointer scenario, where a pointer or reference
outlives the stack variable it refers to.

.. list-table::
   :widths: 50 50
   :header-rows: 1
   :class: colored-code-table

   * - Use after scope
     - Correct
   * -
       .. code-block:: c++

         void foo() {
           int* p;
           {
             int i = 0;
             p = &i;  // warning: 'p' does not live long enough
           }          // note: destroyed here
           (void)*p;  // note: later used here
         }
     -
       .. code-block:: c++

         void foo() {
           int i = 0;
           int* p;
           {
             p = &i; // OK!
           }
           (void)*p;
         }

Return of stack address
-----------------------

This check warns when a function returns a pointer or reference to a
stack-allocated variable, which will be destroyed when the function returns,
leaving the caller with a dangling pointer.◊

.. list-table::
   :widths: 50 50
   :header-rows: 1
   :class: colored-code-table

   * - Return of stack address
     - Correct
   * -
       .. code-block:: c++

        #include <string>
        #include <string_view>

        std::string_view bar() {
          std::string s = "on stack";
          std::string_view result = s;
          // warning: address of stack variable 's' is returned later
          return result; // note: returned here
        }
     -
       .. code-block:: c++

        #include <string>
        #include <string_view>

        std::string bar() {
          std::string s = "on stack";
          std::string_view result = s;
          return result; // OK!
        }


Dangling field
--------------

This check warns when a constructor or method assigns a pointer to a
stack-allocated variable or temporary to a field of the class, and the
stack variable's lifetime is shorter than the object's lifetime.

.. list-table::
   :widths: 50 50
   :header-rows: 1
   :class: colored-code-table


   * - Dangling field
     - Correct
   * -
       .. code-block:: c++

          #include <string>
          #include <string_view>

          // Constructor finishes, leaving 'field' dangling.
          struct DanglingField {
            std::string_view field; // note: this field dangles
            DanglingField(std::string s) {
              field = s; // warning: stack variable 's' escapes to a field
            }
          };
     -
       .. code-block:: c++

          // Make the field an owner.
          struct DanglingField {
            std::string field;
            DanglingField(std::string s) {
              field = s;
            }
          };
          // Or take a string_view parameter.
          struct DanglingField {
            std::string_view field;
            DanglingField(std::string_view s [[clang::lifetimebound]]) {
              field = s;
            }
          };
         };


Use after invalidation (experimental)
-------------------------------------

This check warns when a reference to a container element (such as an iterator,
pointer or reference) is used after a container operation that may have
invalidated it. For example, adding elements to ``std::vector`` may cause
reallocation, invalidating all existing iterators, pointers and references to
its elements.

.. note::
  Container invalidation checking is highly experimental and may produce false
  positives or miss some invalidations. Field-sensitivity is also limited.

.. list-table::
   :widths: 50 50
   :header-rows: 1
   :class: colored-code-table


   * - Use after invalidation (experimental)
     - Correct
   * -
       .. code-block:: c++

        #include <vector>

        void baz(std::vector<int>& v) {
          int* p = &v[0]; // warning: 'v' is later invalidated
          v.push_back(4); // note: invalidated here
          *p = 10;        // note: later used here
        }
     -
       .. code-block:: c++

        #include <vector>

        void baz(std::vector<int>& v) {
          v.push_back(4);
          int* p = &v[0]; // OK!
          *p = 10;
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

  // The analysis will suggest adding [[clang::lifetimebound]] to 'a'.
  std::string_view return_view(std::string_view a) { 
                            // ^^^^^^^^^^^^^^^^^^
                            // warning: parameter 'a' should be marked [[clang::lifetimebound]]
    return a;               // note: param returned here
  }

Translation-Unit-Wide Analysis and Inference
--------------------------------------------

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

*  ``-Wlifetime-safety-all``: Enables all lifetime safety warnings, including
    dangling pointer checks, annotation suggestions, and annotation validations.

*  ``-Wlifetime-safety``: Enables dangling pointer checks from both the ``permissive`` and ``strict`` groups listed below.

  * ``-Wlifetime-safety-permissive``: Enables high-confidence checks for dangling pointers. **Recommended for initial adoption.**

    * ``-Wlifetime-safety-use-after-scope``: Warns when a pointer to a stack variable is used after the variable's lifetime has ended.
    * ``-Wlifetime-safety-return-stack-addr``: Warns when a function returns a pointer or reference to one of its local stack variables.
    * ``-Wlifetime-safety-dangling-field``: Warns when a class field is assigned a pointer to a temporary or stack variable whose lifetime is shorter than the class instance.
  
  * ``-Wlifetime-safety-strict``: Enables stricter and experimental checks. These may produce false positives in code that uses move semantics heavily, as the analysis might conservatively assume a use-after-free even if ownership was transferred.

    *   ``-Wlifetime-safety-use-after-scope-moved``: Same as ``-Wlifetime-safety-use-after-scope`` but for cases where the variable may have been moved from before its destruction.
    *   ``-Wlifetime-safety-return-stack-addr-moved``: Same as ``-Wlifetime-safety-return-stack-addr`` but for cases where the variable may have been moved from.
    *   ``-Wlifetime-safety-dangling-field-moved``: Same as ``-Wlifetime-safety-dangling-field`` but for cases where the variable may have been moved from.
    *   ``-Wlifetime-safety-invalidation``: Warns when a container iterator or reference to an element is used after an operation that may invalidate it (Experimental).

*   ``-Wlifetime-safety-suggestions``: Enables suggestions to add ``[[clang::lifetimebound]]`` to function parameters and ``this`` parameters.

  * ``-Wlifetime-safety-intra-tu-suggestions``: Suggestions for functions local to the translation unit.
  * ``-Wlifetime-safety-cross-tu-suggestions``: Suggestions for functions visible across translation units (e.g., in headers).

* ``-Wlifetime-safety-validations``: Enables checks that validate existing lifetime annotations.

  * ``-Wlifetime-safety-noescape``: Warns when a parameter marked with ``[[clang::noescape]]`` escapes the function.

Limitations
===========

Move Semantics
--------------
The analysis does not currently track ownership transfers through move operations.
Instead, it uses scope-based lifetime tracking: when an owner goes out of scope,
the analysis assumes the resource is destroyed, even if ownership was transferred
via ``std::move()`` or ``std::unique_ptr::release()``.

This means that if a pointer or view is created from an owner, and that owner is
later moved-from and goes out of scope, the analysis will issue a
``-Wlifetime-safety-*-moved`` warning. This warning indicates that the pointer
may be dangling, even though the resource may still be alive under a new owner.
These are often false positives when ownership has been safely transferred.

To avoid these warnings and ensure correctness, follow the
**"move-first-then-alias"** pattern: create views or raw pointers *after* the
ownership transfer, sourcing them from the new owner rather than the original
owner that will go out of scope.

For example:

.. list-table::
   :widths: 50 50
   :header-rows: 1
   :align: left
   :class: colored-code-table

   * - Anti-Pattern: Aliasing Before Move
     - Good Practice: Move-First-Then-Alias
   * -
       .. code-block:: c++

         #include <memory>

         void use(int*);

         void bar() {
           std::unique_ptr<int> b;
           int* p;
           {
             auto a = std::make_unique<int>(42);
             p = a.get(); // warning!
             b = std::move(a);
           }
           use(p);
         }
     -
       .. code-block:: c++

         #include <memory>

         void use(int*);

         void bar() {
           std::unique_ptr<int> b;
           int* p;
           {
             auto a = std::make_unique<int>(42);
             b = std::move(a);
             p = b.get(); // OK!
           }
           use(p);
         }

The same principle applies when moving ownership using ``std::unique_ptr::release()``:

.. code-block:: c++
  :class: bad-code

  #include <memory>
  #include <utility>

  void use(int*);
  void take_ownership(int*);

  void test_aliasing_before_release() {
    int* p;
    {
      auto u = std::make_unique<int>(1);
      p = u.get();
      //  ^ warning: 'u' does not live long enough!
      take_ownership(u.release());
    } 
    use(p);  
  }

``std::unique_ptr`` with custom deleters
----------------------------------------
The analysis assumes standard ownership semantics for owner types like
``std::unique_ptr``: when a ``unique_ptr`` goes out of scope, it is assumed
that the owned object is destroyed and its memory is deallocated.
However, ``std::unique_ptr`` can be used with a custom deleter that modifies
this behavior. For example, a custom deleter might keep the memory alive
by transferring it to a memory pool, or simply do nothing, allowing
another system to manage the lifetime.

Because the analysis relies on scope-based lifetime for owners, it does not
support custom deleters that extend the lifetime of the owned object beyond
the lifetime of the ``std::unique_ptr``. In such cases, the analysis will
assume the object is destroyed when the ``std::unique_ptr`` goes out of scope,
leading to false positive warnings if pointers to the object are used afterward.

.. code-block:: c++

  #include <memory>

  void use(int*);

  struct NoOpDeleter {
    void operator()(int* p) const {
      // Do not delete p, memory is managed elsewhere.
    }
  };

  void test_custom_deleter() {
    int* p;
    {
      std::unique_ptr<int, NoOpDeleter> u(new int(42));
      p = u.get();  // warning: object whose reference is captured does not live long enough
    }               // note: destroyed here
    // With NoOpDeleter, p would still be valid here.
    // But analysis assumes standard unique_ptr semantics and memory being freed.
    use(p);         // note: later used here
  }

Dangling Fields
---------------
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
    std::string_view name_; // note: this field dangles
    MyWidget(std::string name) : name_(name) {} // warning: address of stack memory escapes to a field
    const char* data() { return name_.data(); } // Potential use-after-free if called
  };

In this case, ``name_`` dangles after the constructor finishes.
Even if ``data()`` is never called, the analysis flags the dangling assignment
in the constructor because it represents a latent bug.
The recommended approach is to ensure fields only point to objects that outlive
the field itself, for example by storing an owned object (e.g., ``std::string``)
or ensuring the borrowed object (e.g., one passed by ``const&``) has a
sufficient lifetime.


Performance
===========

Lifetime analysis relies on Clang's CFG (Control Flow Graph). For functions
with very large or complex CFGs, analysis time can sometimes be significant. To mitigate
this, the analysis allows to skip functions where the number of CFG blocks exceeds
a certain threshold, controlled by the ``-flifetime-safety-max-cfg-blocks=N`` language
option.
