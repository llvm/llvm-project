.. title:: clang-tidy - cppcoreguidelines-rvalue-reference-param-not-moved

cppcoreguidelines-rvalue-reference-param-not-moved
==================================================

Warns when an rvalue reference function parameter is never moved within
the function body.

Rvalue reference parameters indicate a parameter that should be moved with
``std::move`` from within the function body. Any such parameter that is
never moved is confusing and potentially indicative of a buggy program.

Example:

.. code-block:: c++

  void logic(std::string&& Input) {
    std::string Copy(Input); // Oops - forgot to std::move
  }

Note that parameters that are unused and marked as such will not be diagnosed.

Example:

.. code-block:: c++

  void conditional_use([[maybe_unused]] std::string&& Input) {
    // No diagnostic here since Input is unused and marked as such
  }

Options
-------

.. option:: AllowPartialMove

   If set to `true`, the check accepts ``std::move`` calls containing any
   subexpression containing the parameter. CppCoreGuideline F.18 officially
   mandates that the parameter itself must be moved. Default is `false`.

  .. code-block:: c++

    // 'p' is flagged by this check if and only if AllowPartialMove is false
    void move_members_of(pair<Obj, Obj>&& p) {
      pair<Obj, Obj> other;
      other.first = std::move(p.first);
      other.second = std::move(p.second);
    }

    // 'p' is never flagged by this check
    void move_whole_pair(pair<Obj, Obj>&& p) {
      pair<Obj, Obj> other = std::move(p);
    }

.. option:: IgnoreUnnamedParams

   If set to `true`, the check ignores unnamed rvalue reference parameters.
   Default is `false`.

.. option:: IgnoreNonDeducedTemplateTypes

   If set to `true`, the check ignores non-deduced template type rvalue
   reference parameters. Default is `false`.

  .. code-block:: c++

    template <class T>
    struct SomeClass {
      // Below, 'T' is not deduced and 'T&&' is an rvalue reference type.
      // This will be flagged if and only if IgnoreNonDeducedTemplateTypes is
      // false. One suggested fix would be to specialize the class for 'T' and
      // 'T&' separately (e.g., see std::future), or allow only one of 'T' or
      // 'T&' instantiations of SomeClass (e.g., see std::optional).
      SomeClass(T&& t) { }
    };

    // Never flagged, since 'T' is a forwarding reference in a deduced context
    template <class T>
    void forwarding_ref(T&& t) {
      T other = std::forward<T>(t);
    }

This check implements `F.18
<http://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#f18-for-will-move-from-parameters-pass-by-x-and-stdmove-the-parameter>`_
from the C++ Core Guidelines.

