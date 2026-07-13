=========================
C++ Type Aware Allocators
=========================

.. contents::
   :local:

Introduction
============

Clang includes an implementation of P2719 "Type-aware allocation and deallocation
functions".

This is a feature that extends the semantics of `new`, `new[]`, `delete` and
`delete[]` operators to expose the type being allocated to the operator. This
can be used to customize allocation of types without needing to modify the
type declaration, or via template definitions fully generic type aware
allocators.

P2719 introduces a type-identity tag as valid parameter type for all allocation
operators. This tag is a default initialized value of type `std::type_identity<T>`
where T is the type being allocated or deallocated.  Unlike the other placement
arguments this tag is passed as the first parameter to the operator.

The most basic use case is as follows

.. code-block:: c++

  #include <new>
  #include <type_traits>

  struct S {
   // ...
  };

  void *operator new(std::type_identity<S>, size_t, std::align_val_t);
  void operator delete(std::type_identity<S>, void *, size_t, std::align_val_t);

  void f() {
    S *s = new S; // calls ::operator new(std::type_identity<S>(), sizeof(S), alignof(S))
    delete s; // calls ::operator delete(std::type_identity<S>(), s, sizeof(S), alignof(S))
  }

While this functionality alone is powerful and useful, the true power comes
by using templates. In addition to adding the type-identity tag, P2719 allows
the tag parameter to be a dependent specialization of `std::type_identity`,
updates the overload resolution rules to support full template deduction and
constraint semantics, and updates the definition of usual deallocation functions
to include `operator delete` definitions that are templatized on the
type-identity tag.

This allows arbitrarily constrained definitions of the operators that resolve
as would be expected for any other template function resolution, e.g (only
showing `operator new` for brevity)

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

Any declaration of a type aware operator new or operator delete must include a
matching complementary operator defined in the same scope.

Notes
=====

Unconstrained Global Operators
------------------------------

Declaring an unconstrained type aware global operator `new` or `delete` (or
`[]` variants) creates numerous hazards, similar to, but different from, those
created by attempting to replace the non-type aware global operators. For that
reason unconstrained operators are strongly discouraged.

Mismatching Constraints
-----------------------

When declaring global type aware operators you should ensure the constraints
applied to new and delete match exactly, and declare them together. This
limits the risk of having mismatching operators selected due to differing
constraints resulting in changes to prioritization when determining the most
viable candidate.

Declarations Across Libraries
-----------------------------

Declaring a typed allocator for a type in a separate TU or library creates
similar hazards as different libraries and TUs may see (or select) different
definitions.

Under this model something like this would be risky

.. code-block:: c++

  template<typename T>
  void *operator new(std::type_identity<std::vector<T>>, size_t, std::align_val_t);

However this hazard is not present simply due to the use of the a type from
another library:

.. code-block:: c++

  template<typename T>
  struct MyType {
    T thing;
  };
  template<typename T>
  void *operator new(std::type_identity<MyType<std::vector<T>>>, size_t, std::align_val_t);

Here we see `std::vector` being used, but that is not the actual type being
allocated.

Implicit and Placement Parameters
---------------------------------

Type aware allocators are always passed both the implicit alignment and size
parameters in all cases. Explicit placement parameters are supported after the
mandatory implicit parameters.

Publication
===========

`Type-aware allocation and deallocation functions <https://wg21.link/P2719>`_.
Louis Dionne, Oliver Hunt.
