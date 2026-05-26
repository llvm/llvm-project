.. title:: clang-tidy - modernize-use-shared-ptr-array

modernize-use-shared-ptr-array
==============================

Finds ``std::shared_ptr<T>`` constructions that manage dynamically allocated
arrays using explicit array deleters and suggests replacing them with
``std::shared_ptr<T[]>``, which has been available since C++17.

Fix-its are provided when the transformation can be applied safely. Some
patterns are diagnosed without a fix-it (warn-only), and others produce
no diagnostic at all.

The check requires C++17 or later.

Recognized Deleter Forms
------------------------

The following deleter forms trigger the check:

- ``std::default_delete<T[]>()`` and ``std::default_delete<T[]>{}``
- Aliases of ``std::default_delete<T[]>`` introduced via ``using`` or
  ``typedef``
- Qualified, namespace-aliased, and unqualified spellings of
  ``std::default_delete<T[]>``
- Captureless lambdas whose body consists solely of ``delete[] p;``
- Free functions whose body consists solely of ``delete[] p;`` where the
  deleted pointer is the function parameter and the definition is visible
  in the translation unit

Warnings Without Fix-its
^^^^^^^^^^^^^^^^^^^^^^^^

The following patterns are diagnosed but not automatically rewritten.

Multiple declarators sharing one type specifier, because the shared
``TypeLoc`` makes independent insertions unsafe.

Pointer and reference declarators, because rewriting the declared type and
constructor expression independently is not safe.

Assignment expressions, because the declaration site is not reachable for
transformation.

Chained assignments for the same reason.

No Diagnostic
^^^^^^^^^^^^^

The following patterns produce no warning.

Constructions that are already correct.

Mismatched new/delete combinations (``new T`` with an array deleter,
``new T[]`` with a non-array deleter, or ``new T[]`` with no deleter).

Type mismatches between the ``shared_ptr`` element type, allocated type,
or deleter pointee type.

Covariant arrays, where the allocated type is derived from the
``shared_ptr`` element type.

Functor deleters, because the check cannot statically verify arbitrary
callable objects.

Function pointer variables, because only direct ``FunctionDecl`` references
are recognized.

Capturing lambdas.

Lambda bodies that are not solely ``delete[] p;``, including
multi-statement, conditional, or indirect deletion forms.

Free functions that are not solely ``delete[] p;``.

Template-dependent contexts, where element types are not fully resolved.

Calls to ``shared_ptr::reset()``, which are member function calls rather
than constructor expressions.

Constructions where the managed pointer expression or deleter argument
originates inside a macro expansion.
