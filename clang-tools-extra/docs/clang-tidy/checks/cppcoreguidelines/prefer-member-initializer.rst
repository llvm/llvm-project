.. title:: clang-tidy - cppcoreguidelines-prefer-member-initializer

cppcoreguidelines-prefer-member-initializer
===========================================

Finds member initializations in the constructor body which can be  converted
into member initializers of the constructor instead. This not only improves
the readability of the code but also positively affects its performance.
Class-member assignments inside a control statement or following the first
control statement are ignored.

This check implements `C.49
<https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#c49-prefer-initialization-to-assignment-in-constructors>`_
from the C++ Core Guidelines.

If the language version is `C++ 11` or above, the constructor is the default
constructor of the class, the field is not a bitfield (only in case of earlier
language version than `C++ 20`), furthermore the assigned value is a literal,
negated literal or ``enum`` constant then the preferred place of the
initialization is at the class member declaration.

This latter rule is `C.48
<https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#c48-prefer-in-class-initializers-to-member-initializers-in-constructors-for-constant-initializers>`_
from the C++ Core Guidelines.

Please note, that this check does not enforce this latter rule for
initializations already implemented as member initializers. For that purpose
see check `modernize-use-default-member-init <../modernize/use-default-member-init.html>`_.

.. note::

  Enforcement of rule C.48 in this check is deprecated, to be removed in
  :program:`clang-tidy` version 19 (only C.49 will be enforced by this check then).
  Please use `cppcoreguidelines-use-default-member-init <../cppcoreguidelines/use-default-member-init.html>`_
  to enforce rule C.48.

Example 1
---------

.. code-block:: c++

  class C {
    int n;
    int m;
  public:
    C() {
      n = 1; // Literal in default constructor
      if (dice())
        return;
      m = 1;
    }
  };

Here ``n`` can be initialized using a default member initializer, unlike
``m``, as ``m``'s initialization follows a control statement (``if``):

.. code-block:: c++

  class C {
    int n{1};
    int m;
  public:
    C() {
      if (dice())
        return;
      m = 1;
    }

Example 2
---------

.. code-block:: c++

  class C {
    int n;
    int m;
  public:
    C(int nn, int mm) {
      n = nn; // Neither default constructor nor literal
      if (dice())
        return;
      m = mm;
    }
  };

Here ``n`` can be initialized in the constructor initialization list, unlike
``m``, as ``m``'s initialization follows a control statement (``if``):

.. code-block:: c++

  C(int nn, int mm) : n(nn) {
    if (dice())
      return;
    m = mm;
  }

.. option:: UseAssignment

   Note: this option is deprecated, to be removed in :program:`clang-tidy`
   version 19. Please use the `UseAssignment` option from
   `cppcoreguidelines-use-default-member-init <../cppcoreguidelines/use-default-member-init.html>`_
   instead.

   If this option is set to `true` (by default `UseAssignment` from
   `modernize-use-default-member-init
   <../modernize/use-default-member-init.html>`_ will be used),
   the check will initialize members with an assignment.
   In this case the fix of the first example looks like this:

.. code-block:: c++

  class C {
    int n = 1;
    int m;
  public:
    C() {
      if (dice())
        return;
      m = 1;
    }
  };
