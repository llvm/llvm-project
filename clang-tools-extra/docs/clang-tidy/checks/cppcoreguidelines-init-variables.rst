.. title:: clang-tidy - cppcoreguidelines-init-variables

cppcoreguidelines-init-variables
================================

Checks whether there are local variables that are declared without an
initial value. These may lead to unexpected behaviour if there is a
code path that reads the variable before assigning to it.

Only integers, booleans, floats, doubles and pointers are checked. The
fix option initializes all detected values with the value of zero. An
exception is float and double types, which are initialized to NaN.

As an example a function that looks like this:

.. code-block:: c++

   void function() {
     int x;
     char *txt;
     double d;

     // Rest of the function.
   }

Would be rewritten to look like this:

.. code-block:: c++

   #include <math.h>

   void function() {
     int x = 0;
     char *txt = nullptr;
     double d = NAN;

     // Rest of the function.
   }

Options
-------

.. option:: IncludeStyle

   A string specifying which include-style is used, `llvm` or
   `google`. Default is `llvm`.

.. option:: MathHeader

   A string specifying the header to include to get the definition of
   `NAN`. Default is `math.h`.
