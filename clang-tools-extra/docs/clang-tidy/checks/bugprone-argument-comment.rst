.. title:: clang-tidy - bugprone-argument-comment

bugprone-argument-comment
=========================

Checks that argument comments match parameter names.

The check understands argument comments in the form ``/*parameter_name=*/``
that are placed right before the argument.

.. code-block:: c++

  void f(bool foo);

  ...

  f(/*bar=*/true);
  // warning: argument name 'bar' in comment does not match parameter name 'foo'

The check tries to detect typos and suggest automated fixes for them.

Options
-------

.. option:: StrictMode

   When zero (default value), the check will ignore leading and trailing
   underscores and case when comparing names -- otherwise they are taken into
   account.
