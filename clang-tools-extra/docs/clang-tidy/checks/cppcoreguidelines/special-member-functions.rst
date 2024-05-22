.. title:: clang-tidy - cppcoreguidelines-special-member-functions

cppcoreguidelines-special-member-functions
==========================================

The check finds classes where some but not all of the special member functions
are defined.

By default the compiler defines a copy constructor, copy assignment operator,
move constructor, move assignment operator and destructor. The default can be
suppressed by explicit user-definitions. The relationship between which
functions will be suppressed by definitions of other functions is complicated
and it is advised that all five are defaulted or explicitly defined.

Note that defining a function with ``= delete`` is considered to be a
definition.

This check implements `C.21
<https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#Rc-five>`_
from the C++ Core Guidelines.

Options
-------

.. option:: AllowSoleDefaultDtor

   When set to `true` (default is `false`), this check will only trigger on
   destructors if they are defined and not defaulted.

   .. code-block:: c++

     struct A { // This is fine.
       virtual ~A() = default;
     };

     struct B { // This is not fine.
       ~B() {}
     };

     struct C {
       // This is not checked, because the destructor might be defaulted in
       // another translation unit.
       ~C();
     };

.. option:: AllowMissingMoveFunctions

   When set to `true` (default is `false`), this check doesn't flag classes which define no move
   operations at all. It still flags classes which define only one of either
   move constructor or move assignment operator. With this option enabled, the following class won't be flagged:

   .. code-block:: c++

     struct A {
       A(const A&);
       A& operator=(const A&);
       ~A();
     };

.. option:: AllowMissingMoveFunctionsWhenCopyIsDeleted

   When set to `true` (default is `false`), this check doesn't flag classes which define deleted copy
   operations but don't define move operations. This flag is related to Google C++ Style Guide
   https://google.github.io/styleguide/cppguide.html#Copyable_Movable_Types. With this option enabled, the
   following class won't be flagged:

   .. code-block:: c++

     struct A {
       A(const A&) = delete;
       A& operator=(const A&) = delete;
       ~A();
     };
