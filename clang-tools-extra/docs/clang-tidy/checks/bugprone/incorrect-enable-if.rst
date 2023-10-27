.. title:: clang-tidy - bugprone-incorrect-enable-if

bugprone-incorrect-enable-if
============================

Detects incorrect usages of ``std::enable_if`` that don't name the nested
``type`` type.

In C++11 introduced ``std::enable_if`` as a convenient way to leverage SFINAE.
One form of using ``std::enable_if`` is to declare an unnamed template type
parameter with a default type equal to
``typename std::enable_if<condition>::type``. If the author forgets to name
the nested type ``type``, then the code will always consider the candidate
template even if the condition is not met.

Below are some examples of code using ``std::enable_if`` correctly and
incorrect examples that this check flags.

.. code-block:: c++

  template <typename T, typename = typename std::enable_if<T::some_trait>::type>
  void valid_usage() { ... }

  template <typename T, typename = std::enable_if_t<T::some_trait>>
  void valid_usage_with_trait_helpers() { ... }

  // The below code is not a correct application of SFINAE. Even if
  // T::some_trait is not true, the function will still be considered in the
  // set of function candidates. It can either incorrectly select the function
  // when it should not be a candidates, and/or lead to hard compile errors
  // if the body of the template does not compile if the condition is not
  // satisfied.
  template <typename T, typename = std::enable_if<T::some_trait>>
  void invalid_usage() { ... }

  // The tool suggests the following replacement for 'invalid_usage':
  template <typename T, typename = typename std::enable_if<T::some_trait>::type>
  void fixed_invalid_usage() { ... }

C++14 introduced the trait helper ``std::enable_if_t`` which reduces the
likelihood of this error. C++20 introduces constraints, which generally
supersede the use of ``std::enable_if``. See
:doc:`modernize-type-traits <../modernize/type-traits>` for another tool
that will replace ``std::enable_if`` with
``std::enable_if_t``, and see
:doc:`modernize-use-constraints <../modernize/use-constraints>` for another
tool that replaces ``std::enable_if`` with C++20 constraints. Consider these
newer mechanisms where possible.
