.. title:: clang-tidy - misc-non-private-member-variables-in-classes

misc-non-private-member-variables-in-classes
============================================

`cppcoreguidelines-non-private-member-variables-in-classes` redirects here
as an alias for this check.

Finds classes that contain non-static data members in addition to user-declared
non-static member functions and diagnose all data members declared with a
non-``public`` access specifier. The data members should be declared as
``private`` and accessed through member functions instead of exposed to derived
classes or class consumers.

Options
-------

.. option:: IgnoreClassesWithAllMemberVariablesBeingPublic

   Allows to completely ignore classes if **all** the member variables in that
   class a declared with a ``public`` access specifier.

.. option:: IgnorePublicMemberVariables

   Allows to ignore (not diagnose) **all** the member variables declared with
   a ``public`` access specifier.
