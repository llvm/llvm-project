.. title:: clang-tidy - readability-stringview-substr

readability-stringview-substr
==============================

Finds ``string_view substr()`` calls that can be replaced with
clearer alternatives using ``remove_prefix()`` or
``remove_suffix()``.

The check suggests the following transformations:

- ``sv = sv.substr(n)`` -> ``sv.remove_prefix(n)``
- ``sv = sv.substr(0, sv.size()-n)`` ->
  ``sv.remove_suffix(n)``
- ``sv = sv.substr(0, sv.size())`` ->
  Redundant self-copy (removed)
- ``sv1 = sv.substr(0, sv.size())`` ->
  ``sv1 = sv``
- ``sv1 = sv2.substr(0, sv2.size()-n)`` ->
  ``sv1 = sv2; sv1.remove_suffix(n)``

.. note::

   ``remove_prefix(n)`` and ``remove_suffix(n)`` have undefined behavior
   when ``n > size()``. The original ``substr()`` call may silently clamp
   out-of-range arguments instead. Verify that the offset is always within
   bounds before applying the suggested fix.
