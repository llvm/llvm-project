.. title:: clang-tidy - performance-enum-size

performance-enum-size
=====================

Recommends the smallest possible underlying type for an ``enum`` or ``enum``
class based on the range of its enumerators. Analyzes the values of the
enumerators in an ``enum`` or ``enum`` class, including signed values, to
recommend the smallest possible underlying type that can represent all the
values of the ``enum``. The suggested underlying types are the integral types
``std::uint8_t``, ``std::uint16_t``, and ``std::uint32_t`` for unsigned types,
and ``std::int8_t``, ``std::int16_t``, and ``std::int32_t`` for signed types.
Using the suggested underlying types can help reduce the memory footprint of
the program and improve performance in some cases.

For example:

.. code-block:: c++

    // BEFORE
    enum Color {
        RED = -1,
        GREEN = 0,
        BLUE = 1
    };

    std::optional<Color> color_opt;

The `Color` ``enum`` uses the default underlying type, which is ``int`` in this
case, and its enumerators have values of -1, 0, and 1. Additionally, the
``std::optional<Color>`` object uses 8 bytes due to padding (platform
dependent).

.. code-block:: c++

    // AFTER
    enum Color : std:int8_t {
        RED = -1,
        GREEN = 0,
        BLUE = 1
    }

    std::optional<Color> color_opt;


In the revised version of the `Color` ``enum``, the underlying type has been
changed to ``std::int8_t``. The enumerator `RED` has a value of -1, which can
be represented by a signed 8-bit integer.

By using a smaller underlying type, the memory footprint of the `Color`
``enum`` is reduced from 4 bytes to 1 byte. The revised version of the
``std::optional<Color>`` object would only require 2 bytes (due to lack of
padding), since it contains a single byte for the `Color` ``enum`` and a single
byte for the ``bool`` flag that indicates whether the optional value is set.

Reducing the memory footprint of an ``enum`` can have significant benefits in
terms of memory usage and cache performance. However, it's important to
consider the trade-offs and potential impact on code readability and
maintainability.

Enums without enumerators (empty) are excluded from analysis.

Requires C++11 or above.
Does not provide auto-fixes.

Options
-------

.. option:: EnumIgnoreList

    Option is used to ignore certain enum types. It accepts a
    semicolon-separated list of (fully qualified) enum type names or regular
    expressions that match the enum type names. The default value is an empty
    string, which means no enums will be ignored.
