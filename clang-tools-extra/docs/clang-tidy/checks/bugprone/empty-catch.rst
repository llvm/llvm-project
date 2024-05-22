.. title:: clang-tidy - bugprone-empty-catch

bugprone-empty-catch
====================

Detects and suggests addressing issues with empty catch statements.

.. code-block:: c++

  try {
    // Some code that can throw an exception
  } catch(const std::exception&) {
  }

Having empty catch statements in a codebase can be a serious problem that
developers should be aware of. Catch statements are used to handle exceptions
that are thrown during program execution. When an exception is thrown, the
program jumps to the nearest catch statement that matches the type of the
exception.

Empty catch statements, also known as "swallowing" exceptions, catch the
exception but do nothing with it. This means that the exception is not handled
properly, and the program continues to run as if nothing happened. This can
lead to several issues, such as:

* *Hidden Bugs*: If an exception is caught and ignored, it can lead to hidden
  bugs that are difficult to diagnose and fix. The root cause of the problem
  may not be apparent, and the program may continue to behave in unexpected
  ways.

* *Security Issues*: Ignoring exceptions can lead to security issues, such as
  buffer overflows or null pointer dereferences. Hackers can exploit these
  vulnerabilities to gain access to sensitive data or execute malicious code.

* *Poor Code Quality*: Empty catch statements can indicate poor code quality
  and a lack of attention to detail. This can make the codebase difficult to
  maintain and update, leading to longer development cycles and increased
  costs.

* *Unreliable Code*: Code that ignores exceptions is often unreliable and can
  lead to unpredictable behavior. This can cause frustration for users and
  erode trust in the software.

To avoid these issues, developers should always handle exceptions properly.
This means either fixing the underlying issue that caused the exception or
propagating the exception up the call stack to a higher-level handler.
If an exception is not important, it should still be logged or reported in
some way so that it can be tracked and addressed later.

If the exception is something that can be handled locally, then it should be
handled within the catch block. This could involve logging the exception or
taking other appropriate action to ensure that the exception is not ignored.

Here is an example:

.. code-block:: c++

  try {
    // Some code that can throw an exception
  } catch (const std::exception& ex) {
    // Properly handle the exception, e.g.:
    std::cerr << "Exception caught: " << ex.what() << std::endl;
  }

If the exception cannot be handled locally and needs to be propagated up the
call stack, it should be re-thrown or new exception should be thrown.

Here is an example:

.. code-block:: c++

  try {
    // Some code that can throw an exception
  } catch (const std::exception& ex) {
    // Re-throw the exception
    throw;
  }

In some cases, catching the exception at this level may not be necessary, and
it may be appropriate to let the exception propagate up the call stack.
This can be done simply by not using ``try/catch`` block.

Here is an example:

.. code-block:: c++

  void function() {
    // Some code that can throw an exception
  }

  void callerFunction() {
    try {
      function();
    } catch (const std::exception& ex) {
      // Handling exception on higher level
      std::cerr << "Exception caught: " << ex.what() << std::endl;
    }
  }

Other potential solution to avoid empty catch statements is to modify the code
to avoid throwing the exception in the first place. This can be achieved by
using a different API, checking for error conditions beforehand, or handling
errors in a different way that does not involve exceptions. By eliminating the
need for try-catch blocks, the code becomes simpler and less error-prone.

Here is an example:

.. code-block:: c++

  // Old code:
  try {
    mapContainer["Key"].callFunction();
  } catch(const std::out_of_range&) {
  }

  // New code
  if (auto it = mapContainer.find("Key"); it != mapContainer.end()) {
    it->second.callFunction();
  }

In conclusion, empty catch statements are a bad practice that can lead to hidden
bugs, security issues, poor code quality, and unreliable code. By handling
exceptions properly, developers can ensure that their code is robust, secure,
and maintainable.

Options
-------

.. option:: IgnoreCatchWithKeywords

  This option can be used to ignore specific catch statements containing
  certain keywords. If a ``catch`` statement body contains (case-insensitive)
  any of the keywords listed in this semicolon-separated option, then the
  catch will be ignored, and no warning will be raised.
  Default value: `@TODO;@FIXME`.

.. option:: AllowEmptyCatchForExceptions

  This option can be used to ignore empty catch statements for specific
  exception types. By default, the check will raise a warning if an empty
  catch statement is detected, regardless of the type of exception being
  caught. However, in certain situations, such as when a developer wants to
  intentionally ignore certain exceptions or handle them in a different way,
  it may be desirable to allow empty catch statements for specific exception
  types.
  To configure this option, a semicolon-separated list of exception type names
  should be provided. If an exception type name in the list is caught in an
  empty catch statement, no warning will be raised.
  Default value: empty string.
