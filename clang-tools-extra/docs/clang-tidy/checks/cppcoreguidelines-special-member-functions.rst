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

This rule is part of the "Constructors, assignments, and destructors" profile of the C++ Core
Guidelines, corresponding to rule C.21. See

https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#c21-if-you-define-or-delete-any-default-operation-define-or-delete-them-all.

Options
-------

.. option:: AllowSoleDefaultDtor

   When set to `true` (default is `false`), this check doesn't flag classes with a sole, explicitly
   defaulted destructor. An example for such a class is:
   
   .. code-block:: c++
   
     struct A {
       virtual ~A() = default;
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
