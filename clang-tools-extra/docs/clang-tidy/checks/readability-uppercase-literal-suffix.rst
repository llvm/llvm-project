.. title:: clang-tidy - readability-uppercase-literal-suffix

readability-uppercase-literal-suffix
====================================

`cert-dcl16-c` redirects here as an alias for this check.
By default, only the suffixes that begin with ``l`` (``l``, ``ll``, ``lu``,
``llu``, but not ``u``, ``ul``, ``ull``) are diagnosed by that alias.

`hicpp-uppercase-literal-suffix` redirects here as an alias for this check.

Detects when the integral literal or floating point (decimal or hexadecimal)
literal has a non-uppercase suffix and provides a fix-it hint with the uppercase
suffix.

All valid combinations of suffixes are supported.

.. code:: c

  auto x = 1;  // OK, no suffix.

  auto x = 1u; // warning: integer literal suffix 'u' is not upper-case

  auto x = 1U; // OK, suffix is uppercase.

  ...

Options
-------

.. option:: NewSuffixes

  Optionally, a list of the destination suffixes can be provided. When the
  suffix is found, a case-insensitive lookup in that list is made, and if a
  replacement is found that is different from the current suffix, then the
  diagnostic is issued. This allows for fine-grained control of what suffixes to
  consider and what their replacements should be.

Example
^^^^^^^

Given a list `L;uL`:

* ``l`` -> ``L``
* ``L`` will be kept as is.
* ``ul`` -> ``uL``
* ``Ul`` -> ``uL``
* ``UL`` -> ``uL``
* ``uL`` will be kept as is.
* ``ull`` will be kept as is, since it is not in the list
* and so on.

.. option:: IgnoreMacros

   If this option is set to non-zero (default is `1`), the check will not warn
   about literal suffixes inside macros.
