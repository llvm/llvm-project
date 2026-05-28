.. title:: clang-tidy - performance-expensive-value-or

performance-expensive-value-or
==============================

Finds calls to ``value_or`` on optional types where the underlying value type
is expensive to copy. While ``value()`` and ``operator*`` return references,
``value_or`` always returns by value, which involves copying the contained
value.

The check is applied to types that are not trivially copyable or whose size
exceeds a configurable threshold. It supports ``std::optional``,
``boost::optional``, ``absl::optional``, and other optional-like types via
configuration.

Example:

.. code-block:: c++

    #include <optional>
    #include <string>

    void example(std::optional<std::string> opt) {
      // Warning: copies the std::string out of the optional.
      auto val = opt.value_or("default");

      // Alternatives that avoid the copy:
      const std::string fallback = "default";
      const auto &ref = opt.has_value() ? *opt : fallback;
    }

Options
-------

.. option:: SizeThreshold

   The minimum size in bytes (exclusive) above which a trivially-copyable type
   is considered expensive to copy. Types with ``sizeof(T) > SizeThreshold``
   trigger the warning even if they are trivially copyable. Types at or below
   this threshold only trigger if they are not trivially copyable.
   Default is `8`.

.. option:: OptionalTypes

   Semicolon-separated list of fully-qualified names of optional-like class
   templates to check. The check matches calls to ``value_or`` on
   specializations of these templates. Default is `::std::optional`.

   Example configuration to also check ``absl::optional``:

   .. code-block:: yaml

       CheckOptions:
         performance-expensive-value-or.OptionalTypes: "::std::optional;::absl::optional"

.. option:: WarnOnRvalueOptional

   When `false` (default), the check does not warn when ``value_or`` is called
   on an rvalue optional (e.g., a function return value). The rationale is that
   rvalue overloads of ``value_or`` may move instead of copy, and replacing the
   call would require materializing the temporary into a named variable to check
   it before dereferencing. Set to `true` to warn in all cases.
