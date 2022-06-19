.. title:: clang-tidy - readability-duplicate-include

readability-duplicate-include
=============================

Looks for duplicate includes and removes them.  The check maintains a list of
included files and looks for duplicates.  If a macro is defined or undefined
then the list of included files is cleared.

Examples:

.. code-block:: c++

  #include <memory>
  #include <vector>
  #include <memory>

becomes

.. code-block:: c++

  #include <memory>
  #include <vector>

Because of the intervening macro definitions, this code remains unchanged:

.. code-block:: c++

  #undef NDEBUG
  #include "assertion.h"
  // ...code with assertions enabled

  #define NDEBUG
  #include "assertion.h"
  // ...code with assertions disabled
