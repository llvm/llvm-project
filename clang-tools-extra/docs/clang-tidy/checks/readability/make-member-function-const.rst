.. title:: clang-tidy - readability-make-member-function-const

readability-make-member-function-const
======================================

Finds non-static member functions that can be made ``const``
because the functions don't use ``this`` in a non-const way.

This check tries to annotate methods according to
`logical constness <https://isocpp.org/wiki/faq/const-correctness#logical-vs-physical-state>`_
(not physical constness).
Therefore, it will suggest to add a ``const`` qualifier to a non-const
method only if this method does something that is already possible though the
public interface on a ``const`` pointer to the object:

* reading a public member variable
* calling a public const-qualified member function
* returning const-qualified ``this``
* passing const-qualified ``this`` as a parameter.

This check will also suggest to add a ``const`` qualifier to a non-const
method if this method uses private data and functions in a limited number of
ways where logical constness and physical constness coincide:

* reading a member variable of builtin type

Specifically, this check will not suggest to add a ``const`` to a non-const
method if the method reads a private member variable of pointer type because
that allows to modify the pointee which might not preserve logical constness.
For the same reason, it does not allow to call private member functions
or member functions on private member variables.

In addition, this check ignores functions that

* are declared ``virtual``
* contain a ``const_cast``
* are templated or part of a class template
* have an empty body
* do not (implicitly) use ``this`` at all
  (see `readability-convert-member-functions-to-static <../readability/convert-member-functions-to-static.html>`_).

The following real-world examples will be preserved by the check:

.. code-block:: c++

  class E1 {
    Pimpl &getPimpl() const;
  public:
    int &get() {
      // Calling a private member function disables this check.
      return getPimpl()->i;
    }
    ...
  };

  class E2 {
  public:
    const int *get() const;
    // const_cast disables this check.
    S *get() {
      return const_cast<int*>(const_cast<const C*>(this)->get());
    }
    ...
  };

After applying modifications as suggested by the check, running the check again
might find more opportunities to mark member functions ``const``.
