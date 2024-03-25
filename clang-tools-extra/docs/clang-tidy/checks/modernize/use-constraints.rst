.. title:: clang-tidy - modernize-use-constraints

modernize-use-constraints
=========================

Replace ``std::enable_if`` with C++20 requires clauses.

``std::enable_if`` is a SFINAE mechanism for selecting the desired function or
class template based on type traits or other requirements. ``enable_if`` changes
the meta-arity of the template, and has other
`adverse side effects
<https://open-std.org/JTC1/SC22/WG21/docs/papers/2016/p0225r0.html>`_
in the code. C++20 introduces concepts and constraints as a cleaner language
provided solution to achieve the same outcome.

This check finds some common ``std::enable_if`` patterns that can be replaced
by C++20 requires clauses. The tool can replace some of these patterns
automatically, otherwise, the tool will emit a diagnostic without a
replacement. The tool can detect the following ``std::enable_if`` patterns

1. ``std::enable_if`` in the return type of a function
2. ``std::enable_if`` as the trailing template parameter for function templates

Other uses, for example, in class templates for function parameters, are not
currently supported by this tool. Other variants such as ``boost::enable_if``
are not currently supported by this tool.

Below are some examples of code using ``std::enable_if``.

.. code-block:: c++

  // enable_if in function return type
  template <typename T>
  std::enable_if_t<T::some_trait, int> only_if_t_has_the_trait() { ... }

  // enable_if in the trailing template parameter
  template <typename T, std::enable_if_t<T::some_trait, int> = 0>
  void another_version() { ... }

  template <typename T>
  typename std::enable_if<T::some_value, Obj>::type existing_constraint() requires (T::another_value) {
    return Obj{};
  }

  template <typename T, std::enable_if_t<T::some_trait, int> = 0>
  struct my_class {};

The tool will replace the above code with,

.. code-block:: c++

  // warning: use C++20 requires constraints instead of enable_if [modernize-use-constraints]
  template <typename T>
  int only_if_t_has_the_trait() requires T::some_trait { ... }

  // warning: use C++20 requires constraints instead of enable_if [modernize-use-constraints]
  template <typename T>
  void another_version() requires T::some_trait { ... }

  // The tool will emit a diagnostic for the following, but will
  // not attempt to replace the code.
  // warning: use C++20 requires constraints instead of enable_if [modernize-use-constraints]
  template <typename T>
  typename std::enable_if<T::some_value, Obj>::type existing_constraint() requires (T::another_value) {
    return Obj{};
  }

  // The tool will not emit a diagnostic or attempt to replace the code.
  template <typename T, std::enable_if_t<T::some_trait, int> = 0>
  struct my_class {};
