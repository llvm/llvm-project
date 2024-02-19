.. title:: clang-tidy - bugprone-optional-value-conversion

bugprone-optional-value-conversion
==================================

Detects potentially unintentional and redundant conversions where a value is
extracted from an optional-like type and then used to create a new instance of
the same optional-like type.

These conversions might be the result of developer oversight, leftovers from
code refactoring, or other situations that could lead to unintended exceptions
or cases where the resulting optional is always initialized, which might be
unexpected behavior.

To illustrate, consider the following problematic code snippet:

.. code-block:: c++

    #include <optional>

    void print(std::optional<int>);

    int main()
    {
      std::optional<int> opt;
      // ...

      // Unintentional conversion from std::optional<int> to int and back to
      // std::optional<int>:
      print(opt.value());

      // ...
    }

A better approach would be to directly pass ``opt`` to the ``print`` function
without extracting its value:

.. code-block:: c++

    #include <optional>

    void print(std::optional<int>);

    int main()
    {
      std::optional<int> opt;
      // ...

      // Proposed code: Directly pass the std::optional<int> to the print
      // function.
      print(opt);

      // ...
    }

By passing ``opt`` directly to the print function, unnecessary conversions are
avoided, and potential unintended behavior or exceptions are minimized.

Value extraction using ``operator *`` is matched by default.
The support for non-standard optional types such as ``boost::optional`` or
``absl::optional`` may be limited.

Options:
--------

.. option:: OptionalTypes

    Semicolon-separated list of (fully qualified) optional type names or regular
    expressions that match the optional types.
    Default value is `::std::optional;::absl::optional;::boost::optional`.

.. option:: ValueMethods

    Semicolon-separated list of (fully qualified) method names or regular
    expressions that match the methods.
    Default value is `::value$;::get$`.
