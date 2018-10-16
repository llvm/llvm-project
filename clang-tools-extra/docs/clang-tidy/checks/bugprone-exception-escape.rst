.. title:: clang-tidy - bugprone-exception-escape

bugprone-exception-escape
=========================

Finds functions which may throw an exception directly or indirectly, but they
should not. The functions which should not throw exceptions are the following:
* Destructors
* Move constructors
* Move assignment operators
* The ``main()`` functions
* ``swap()`` functions
* Functions marked with ``throw()`` or ``noexcept``
* Other functions given as option

A destructor throwing an exception may result in undefined behavior, resource
leaks or unexpected termination of the program. Throwing move constructor or
move assignment also may result in undefined behavior or resource leak. The
``swap()`` operations expected to be non throwing most of the cases and they
are always possible to implement in a non throwing way. Non throwing ``swap()``
operations are also used to create move operations. A throwing ``main()``
function also results in unexpected termination.

WARNING! This check may be expensive on large source files.

Options
-------

.. option:: FunctionsThatShouldNotThrow

   Comma separated list containing function names which should not throw. An
   example value for this parameter can be ``WinMain`` which adds function
   ``WinMain()`` in the Windows API to the list of the funcions which should
   not throw. Default value is an empty string.

.. option:: IgnoredExceptions

   Comma separated list containing type names which are not counted as thrown
   exceptions in the check. Default value is an empty string.
