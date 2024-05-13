.. title:: clang-tidy - readability-identifier-length

readability-identifier-length
=============================

This check finds variables and function parameters whose length are too short.
The desired name length is configurable.

Special cases are supported for loop counters and for exception variable names.

Options
-------

The following options are described below:

 - :option:`MinimumVariableNameLength`, :option:`IgnoredVariableNames`
 - :option:`MinimumParameterNameLength`, :option:`IgnoredParameterNames`
 - :option:`MinimumLoopCounterNameLength`, :option:`IgnoredLoopCounterNames`
 - :option:`MinimumExceptionNameLength`, :option:`IgnoredExceptionVariableNames`

.. option:: MinimumVariableNameLength

    All variables (other than loop counter, exception names and function
    parameters) are expected to have at least a length of
    `MinimumVariableNameLength` (default is `3`). Setting it to `0` or `1`
    disables the check entirely.


    .. code-block:: c++

         int doubler(int x)   // warns that x is too short
         {
            return 2 * x;
         }

    This check does not have any fix suggestions in the general case since
    variable names have semantic value.

.. option:: IgnoredVariableNames

    Specifies a regular expression for variable names that are
    to be ignored. The default value is empty, thus no names are ignored.

.. option:: MinimumParameterNameLength

    All function parameter names are expected to have a length of at least
    `MinimumParameterNameLength` (default is `3`). Setting it to `0` or `1`
    disables the check entirely.


    .. code-block:: c++

      int i = 42;    // warns that 'i' is too short

    This check does not have any fix suggestions in the general case since
    variable names have semantic value.

.. option:: IgnoredParameterNames

    Specifies a regular expression for parameters that are to be ignored.
    The default value is `^[n]$` for historical reasons.

.. option:: MinimumLoopCounterNameLength

    Loop counter variables are expected to have a length of at least
    `MinimumLoopCounterNameLength` characters (default is `2`). Setting it to
    `0` or `1` disables the check entirely.


    .. code-block:: c++

      // This warns that 'q' is too short.
      for (int q = 0; q < size; ++ q) {
         // ...
      }

.. option:: IgnoredLoopCounterNames

    Specifies a regular expression for counter names that are to be ignored.
    The default value is `^[ijk_]$`; the first three symbols for historical
    reasons and the last one since it is frequently used as a "don't care"
    value, specifically in tools such as Google Benchmark.


    .. code-block:: c++

      // This does not warn by default, for historical reasons.
      for (int i = 0; i < size; ++ i) {
          // ...
      }

.. option:: MinimumExceptionNameLength

    Exception clause variables are expected to have a length of at least
    `MinimumExceptionNameLength` (default is `2`). Setting it to `0` or `1`
    disables the check entirely.


    .. code-block:: c++

      try {
          // ...
      }
      // This warns that 'e' is too short.
      catch (const std::exception& x) {
          // ...
      }

.. option:: IgnoredExceptionVariableNames

    Specifies a regular expression for exception variable names that are to
    be ignored. The default value is `^[e]$` mainly for historical reasons.

    .. code-block:: c++

      try {
          // ...
      }
      // This does not warn by default, for historical reasons.
      catch (const std::exception& e) {
          // ...
      }
