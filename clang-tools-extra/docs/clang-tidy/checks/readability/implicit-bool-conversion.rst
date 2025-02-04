.. title:: clang-tidy - readability-implicit-bool-conversion

readability-implicit-bool-conversion
====================================

This check can be used to find implicit conversions between built-in types and
booleans. Depending on use case, it may simply help with readability of the code,
or in some cases, point to potential bugs which remain unnoticed due to implicit
conversions.

The following is a real-world example of bug which was hiding behind implicit
``bool`` conversion:

.. code-block:: c++

  class Foo {
    int m_foo;

  public:
    void setFoo(bool foo) { m_foo = foo; } // warning: implicit conversion bool -> int
    int getFoo() { return m_foo; }
  };

  void use(Foo& foo) {
    bool value = foo.getFoo(); // warning: implicit conversion int -> bool
  }

This code is the result of unsuccessful refactoring, where type of ``m_foo``
changed from ``bool`` to ``int``. The programmer forgot to change all
occurrences of ``bool``, and the remaining code is no longer correct, yet it
still compiles without any visible warnings.

In addition to issuing warnings, fix-it hints are provided to help solve the
reported issues. This can be used for improving readability of code, for
example:

.. code-block:: c++

  void conversionsToBool() {
    float floating;
    bool boolean = floating;
    // ^ propose replacement: bool boolean = floating != 0.0f;

    int integer;
    if (integer) {}
    // ^ propose replacement: if (integer != 0) {}

    int* pointer;
    if (!pointer) {}
    // ^ propose replacement: if (pointer == nullptr) {}

    while (1) {}
    // ^ propose replacement: while (true) {}
  }

  void functionTakingInt(int param);

  void conversionsFromBool() {
    bool boolean;
    functionTakingInt(boolean);
    // ^ propose replacement: functionTakingInt(static_cast<int>(boolean));

    functionTakingInt(true);
    // ^ propose replacement: functionTakingInt(1);
  }

In general, the following conversion types are checked:

- integer expression/literal to boolean (conversion from a single bit bitfield
  to boolean is explicitly allowed, since there's no ambiguity / information
  loss in this case),

- floating expression/literal to boolean,

- pointer/pointer to member/``nullptr``/``NULL`` to boolean,

- boolean expression/literal to integer (conversion from boolean to a single
  bit bitfield is explicitly allowed),

- boolean expression/literal to floating.

The rules for generating fix-it hints are:

- in case of conversions from other built-in type to bool, an explicit
  comparison is proposed to make it clear what exactly is being compared:

  - ``bool boolean = floating;`` is changed to
    ``bool boolean = floating == 0.0f;``,

  - for other types, appropriate literals are used (``0``, ``0u``, ``0.0f``,
    ``0.0``, ``nullptr``),

- in case of negated expressions conversion to bool, the proposed replacement
  with comparison is simplified:

  - ``if (!pointer)`` is changed to ``if (pointer == nullptr)``,

- in case of conversions from bool to other built-in types, an explicit
  ``static_cast`` (or a C-style cast since C23) is proposed to make it clear
  that a conversion is taking place:

  - ``int integer = boolean;`` is changed to
    ``int integer = static_cast<int>(boolean);``,

- if the conversion is performed on type literals, an equivalent literal is
  proposed, according to what type is actually expected, for example:

  - ``functionTakingBool(0);`` is changed to ``functionTakingBool(false);``,

  - ``functionTakingInt(true);`` is changed to ``functionTakingInt(1);``,

  - for other types, appropriate literals are used (``false``, ``true``, ``0``,
    ``1``, ``0u``, ``1u``, ``0.0f``, ``1.0f``, ``0.0``, ``1.0f``).

Some additional accommodations are made for pre-C++11 dialects:

- ``false`` literal conversion to pointer is detected,

- instead of ``nullptr`` literal, ``0`` is proposed as replacement.

Occurrences of implicit conversions inside macros and template instantiations
are deliberately ignored, as it is not clear how to deal with such cases.

Options
-------

.. option::  AllowIntegerConditions

   When `true`, the check will allow conditional integer conversions. Default
   is `false`.

.. option::  AllowPointerConditions

   When `true`, the check will allow conditional pointer conversions. Default
   is `false`.

.. option::  UseUpperCaseLiteralSuffix

   When `true`, the replacements will use an uppercase literal suffix in the
   provided fixes. Default is `false`.

    Example

    .. code-block:: c++

      uint32_t foo;
      if (foo) {}
      // ^ propose replacement default: if (foo != 0u) {}
      // ^ propose replacement with option `UseUpperCaseLiteralSuffix`: if (foo != 0U) {}
