.. title:: clang-tidy - modernize-type-traits

modernize-type-traits
=====================

Converts standard library type traits of the form ``traits<...>::type`` and
``traits<...>::value`` into ``traits_t<...>`` and ``traits_v<...>`` respectively.

For example:

.. code-block:: c++

  std::is_integral<T>::value
  std::is_same<int, float>::value
  typename std::add_const<T>::type
  std::make_signed<unsigned>::type

Would be converted into:

.. code-block:: c++

  std::is_integral_v<T>
  std::is_same_v<int, float>
  std::add_const_t<T>
  std::make_signed_t<unsigned>

Options
-------

.. option:: IgnoreMacros

  If `true` don't diagnose traits defined in macros.

  Note: Fixes will never be emitted for code inside of macros.

  .. code-block:: c++

    #define IS_SIGNED(T) std::is_signed<T>::value

  Defaults to `false`. 
