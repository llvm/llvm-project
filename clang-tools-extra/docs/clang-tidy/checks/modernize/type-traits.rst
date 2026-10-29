.. title:: clang-tidy - modernize-type-traits

modernize-type-traits
=====================

Converts standard library type traits of the form ``traits<...>::type`` and
``traits<...>::value`` into ``traits_t<...>`` and
``traits_v<...>`` respectively.

Also suggests converting ``std::remove_cv_t<std::remove_reference_t<...>`` into
``std::remove_cvref_t<...>`` when targeting C++20 or above.

For example:

.. code-block:: c++

  std::is_integral<T>::value
  std::is_same<int, float>::value
  typename std::add_const<T>::type
  std::make_signed<unsigned>::type

  std::remove_cv_t<std::remove_reference_t<int>>

Would be converted into:

.. code-block:: c++

  std::is_integral_v<T>
  std::is_same_v<int, float>
  std::add_const_t<T>
  std::make_signed_t<unsigned>

  std::remove_cvref_t<int>

Does not diagnose ``std::tuple_size<T>::value`` and
``std::variant_size<T>::value`` inside requires-expressions, concept definitions
and requires-clauses. There, a substitution failure in the ``::value`` form
makes the constraint unsatisfied, while ``std::tuple_size_v<T>`` and
``std::variant_size_v<T>`` would cause a hard error instead.


Options
-------

.. option:: IgnoreMacros

  If `true` don't diagnose traits defined in macros.

  Note: Fixes will never be emitted for code inside of macros.

  .. code-block:: c++

    #define IS_SIGNED(T) std::is_signed<T>::value

  Defaults to `false`.


Limitations
-----------

Does not currently diagnose uses of type traits with nested name
specifiers (e.g. ``std::chrono::is_clock``,
``std::chrono::treat_as_floating_point``).
