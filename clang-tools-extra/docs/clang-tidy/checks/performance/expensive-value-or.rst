.. title:: clang-tidy - performance-expensive-value-or

performance-expensive-value-or
==============================

Finds calls to ``value_or`` (and alternative spellings ``valueOr``,
``ValueOr``) on optional types where the return type is expensive to copy.
These methods return by value, which involves copying the contained value.
``value()`` and ``operator*`` return references and can be used to avoid the
copy when appropriate.

The check is applied to types that are not trivially copyable or whose size
exceeds a configurable threshold. It supports ``std::optional``,
``boost::optional``, ``absl::optional``, and other optional-like types via
configuration.

Example:

.. code-block:: c++

    #include <optional>
    #include <string>

    void consumeRef(const std::string &);

    void example(std::optional<std::string> opt,
                 const std::string &fallback) {
      // Warning: result binds to const reference, copy is avoidable.
      const std::string &ref = opt.value_or(fallback);

      // Warning: result passed to const reference parameter.
      consumeRef(opt.value_or(fallback));

      // Warning: const member called on temporary.
      auto len = opt.value_or(fallback).size();

      // No warning by default: caller takes ownership.
      std::string val = opt.value_or("default");
    }

By default, the check only warns in reference-friendly contexts where the copy
is clearly avoidable: binding to ``const T&``, passing to a ``const T&``
parameter, or calling a const member function on the temporary. Contexts where
the caller takes ownership (binding to a value, passing to a by-value
parameter) are not flagged unless ``WarnOnOwnershipTaking`` is enabled.

Options
-------

.. option:: SizeThreshold

   The minimum size in bytes (exclusive) above which a trivially-copyable type
   is considered expensive to copy. Types with ``sizeof(T) > SizeThreshold``
   trigger the warning even if they are trivially copyable. Types at or below
   this threshold only trigger if they are not trivially copyable.
   Default is `16`.

.. option:: OptionalTypes

   Semicolon-separated list of regular expressions matching fully-qualified
   names of optional-like class templates to check. The check matches calls to
   ``value_or`` on specializations of these templates. Default is
   `::std::optional;::absl::optional;::boost::optional`.

   Example configuration to also check a project-local optional type:

   .. code-block:: yaml

       CheckOptions:
         performance-expensive-value-or.OptionalTypes: "::std::optional;::absl::optional;::boost::optional;::myproject::.*Optional"

.. option:: WarnOnOwnershipTaking

   When `true`, the check also warns when the result of ``value_or`` is used in
   an ownership-taking context (e.g., initializing a value variable or passing
   to a by-value parameter). Default is `false`.
