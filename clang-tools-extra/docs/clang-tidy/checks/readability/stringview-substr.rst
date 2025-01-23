.. title:: clang-tidy - readability-stringview-substr

readability-stringview-substr
==============================

Finds ``string_view substr()`` calls that can be replaced with clearer alternatives
using ``remove_prefix()`` or ``remove_suffix()``.

The check suggests the following transformations:

===========================================  =======================================
Expression                                   Replacement
===========================================  =======================================
``sv = sv.substr(n)``                        ``sv.remove_prefix(n)``
``sv = sv.substr(0, sv.length()-n)``         ``sv.remove_suffix(n)``
``sv = sv.substr(0, sv.length())``           Redundant self-copy
``sv1 = sv.substr(0, sv.length())``          ``sv1 = sv``
``sv1 = sv2.substr(0, sv2.length()-n)``      ``sv1 = sv2;`` ``sv1.remove_suffix(n)``
===========================================  =======================================
