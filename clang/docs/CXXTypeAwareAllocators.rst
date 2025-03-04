=========================
C++ Type Aware Allocators
=========================

.. contents::
   :local:

Introduction
============

Clang includes an experimental implementation of P2719 "Type-aware allocation
and deallocation functions".

This is a feature that extends the semantics of ``new``, ``new[]``, ``delete`` and
``delete[]`` operators to expose the type being allocated to the operator. This
can be used to customize allocation of types without needing to modify the
type declaration, or via template definitions fully generic type aware
allocators.

A major use case of this feature is to support hardened or secure allocators
by supporting anything from simple type property based hardening through to
complete type isolating allocators and beyond, and as such there are no
restrictions on the types or locations that it can be used - anywhere
an allocation or deallocation operator can be declared today can be extended
or replaced with an equivalent type aware declaration.

Beyond security this feature also allows developers to make rules around
how types may be allocated more explicit by controlling the use and
availability of ``new`` and ``delete`` for types without needing to directly
modify the type. This can be useful where allocation is expected to be
performed through specific interfaces, or explicitly via global ``new`` and
``delete`` operators.

P2719 introduces a type-identity tag as valid parameter type for all allocation
operators. This tag is a default initialized value of type 
``std::type_identity<T>`` where T is the type being allocated or deallocated.
Unlike the other placement arguments this tag is passed as the first parameter
to the operator.

Type aware allocation functions default to stricter semantics than non-type
aware variants. The normally optional implicit size and alignment parameters are
mandatory for type aware operators, and are not permitted to be dependent types.
If a cleanup delete cannot be found it is a hard error rather than a silent leak
and it is an error if the `operator new` and `operator delete` functions for a
type are not defined in the same scope.

Usage
=====

Type aware allocation is currently disabled by default, to enable it use the
``-fexperimental-cxx-type-aware-allocators`` argument to clang.

The most basic usage is as follows

.. code-block:: c++

  #include <new>
  #include <type_traits>
  
  struct S {
   // ...
  };
  
  void *operator new(std::type_identity<S>, size_t, std::align_val_t);
  void operator delete(std::type_identity<S>, void *, size_t, std::align_val_t);
  
  void f() {
    S *s = new S; // calls ::operator new(std::type_identity<S>(), sizeof(S), std::align_val_t(alignof(S)))
    delete s; // calls ::operator delete(std::type_identity<S>(), s, sizeof(S), std::align_val_t(alignof(S)))
  }

While this functionality alone is powerful and useful, the true power comes
by using templates. In addition to adding the type-identity tag, P2719 allows
the tag parameter to be a dependent specialization of `std::type_identity`,
updates the overload resolution rules to support full template deduction and
constraint semantics, and updates the definition of usual deallocation functions
to include ``operator delete`` definitions that are templatized on the
type-identity tag.

This allows arbitrarily constrained definitions of the operators that resolve
as would be expected for any other template function resolution, e.g (only
showing ``operator new`` for brevity)

.. code-block:: c++

   template <typename T, unsigned Size> struct Array {
     T buffer[Size];
   };

   // Starting with a concrete type
   void *operator new(std::type_identity<Array<int, 5>>, size_t, std::align_val_t);
   
   // Only care about five element arrays
   template <typename T>
   void *operator new(std::type_identity<Array<T, 5>>, size_t, std::align_val_t);
   
   // An array of N floats
   template <unsigned N>
   void *operator new(std::type_identity<Array<float, N>>, size_t, std::align_val_t);

   // Any array
   template <typename T, unsigned N>
   void *operator new(std::type_identity<Array<T, N>>, size_t, std::align_val_t);

   // A handy concept
   template <typename T> concept Polymorphic = std::is_polymorphic_v<T>;

   // Only applies is T is Polymorphic
   template <Polymorphic T, unsigned N>
   void *operator new(std::type_identity<Array<T, N>>, size_t, std::align_val_t);

   // Any even length array
   template <typename T, unsigned N>
   void *operator new(std::type_identity<Array<T, N>>, size_t, std::align_val_t)
       requires(N%2 == 0);

Operator selection then proceeds according to the usual rules for choosing
the best/most constrained match.

Notes
=====

Class Scoped Operators
----------------------

Class scoped type aware allocation and deallocation operators are permitted,
and should be preferred over global operators with subtyping constraints where
possible, as even with a subtyping constraint it is possible to get
:ref:`mismatching constraints<cxxtypeawareallocators-mismatching-constraint>` or
:ref:`alternate TUs <cxxtypeawareallocators-declarations-across-libraries-and-TUs>`
that result in mismatched operators being selected.

The only restriction is that P2719 does not by default permit type aware
destroying delete. This is due to the semantic complexity that comes from the
type being provided being the static type of the object, not the dynamic type
as the primary use case for which destroying delete exists is when a developer
is avoiding dynamic dispatch.

Subclassing and polymorphism
----------------------------

While a type aware ``operator new`` will always receive the exact type being
allocated, deletion is limited to awareness of the dynamic type of an object.
If deletion is performed via a virtual call, the type-identity tag passed to
the type aware ``operator delete`` will be the dynamic type of the object.

Absent virtual dispatch the type-identity tag provided to operator delete is
subject to the same limitations of object deletion and destruction of
non-type-aware deletion and destruction, where method selection and dispatch
is based solely on the static type of the object at the call site. As such
the received type-identity tag will reflect the static type at the call site,
not the dynamic type of the object being deleted.

Unconstrained Global Operators
------------------------------

Declaring an unconstrained type aware global operator ``new`` or ``delete`` (or
``[]`` variants) creates numerous hazards, similar to, but different from, those
created by attempting to replace the non-type aware global operators. For that
reason unconstrained operators are strongly discouraged.

.. _cxxtypeawareallocators-mismatching-constraint:

Mismatching Constraints
-----------------------

When declaring global type aware operators you should ensure the constraints
applied to new and delete match exactly, and declare them together. This
limits the risk of having mismatching operators selected due to differing
constraints resulting in changes to prioritization when determining the most
viable candidate.

.. _cxxtypeawareallocators-declarations-across-libraries-and-TUs:

Declarations Across Libraries and TUs
-------------------------------------

Declaring a typed allocator for a type in a separate TU or library creates
similar hazards as different libraries and TUs may see (or select) different
definitions.

Under this model something like this would be risky

.. code-block:: c++

  template<typename T>
  void *operator new(std::type_identity<std::vector<T>>, size_t);

However this hazard is not present simply due to the use of the a type from
another library:

.. code-block:: c++

  template<typename T>
  struct MyType {
    T thing;
  };
  template<typename T>
  void *operator new(std::type_identity<MyType<std::vector<T>>>, size_t);

Here we see `std::vector` being used, but that is not the actual type being
allocated.

Implicit and Placement Parameters
---------------------------------

Type aware allocators require the implicit alignment and size (for delete)
parameters, and allow any other explicit placement parameters supported in
non-type aware operators.

Constant Evaluation
-------------------

Type aware allocation functions declared in the global scope are considered
usual deallocation functions if the only difference between the type aware
declaration and a usual deallocation function is the type-identity parameter.
This eases the use of dynamic allocation of types with type aware allocation
functions within constant contexts. Unfortunately this does not resolve the
problem of class-scoped new and delete in constant contexts, as the existence of
such declarations precludes lookup in the global scope and as a result
class-scoped operators still prevents the use of a type in a constant context.

Publication
===========

`Type-aware allocation and deallocation functions <https://wg21.link/P2719>`_.
Louis Dionne, Oliver Hunt.