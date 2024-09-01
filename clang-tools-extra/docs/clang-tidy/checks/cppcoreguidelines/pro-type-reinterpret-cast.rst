.. title:: clang-tidy - cppcoreguidelines-pro-type-reinterpret-cast

cppcoreguidelines-pro-type-reinterpret-cast
===========================================

This check flags all uses of ``reinterpret_cast`` in C++ code.

Use of these casts can violate type safety and cause the program to access a
variable that is actually of type ``X`` to be accessed as if it were of an
unrelated type ``Z``.

This rule is part of the `Type safety (Type.1.1)
<https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#Pro-type-reinterpretcast>`_
profile from the C++ Core Guidelines.

Options
-------

.. option:: AllowCastToBytes

  When this setting is set to `true`, it will not warn when casting an object
  to its byte representation, which is safe according to the C++ Standard.
  The allowed byte types are: ``char``, ``unsigned char`` and ``std::byte``.
  Example:

  .. code-block:: cpp

    float x{};
    auto a = reinterpret_cast<char*>(&x);           // OK
    auto b = reinterpret_cast<unsigned char*>(&x);  // OK
    auto c = reinterpret_cast<std::byte*>(&x);      // OK

  Default value is `false`.
