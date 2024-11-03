Contributing
============

Getting Started
---------------

Please refer to the `LLVM Getting Started Guide
<https://llvm.org/docs/GettingStarted.html>`_ for general information on how to
get started on the LLVM project. A detailed explanation on how to build and
test LLDB can be found in the `build instructions <build.html>`_ and `test
instructions <test.html>`_ respectively.

Contributing to LLDB
--------------------

Please refer to the `LLVM Developer Policy
<https://llvm.org/docs/DeveloperPolicy.html>`_ for information about
authoring and uploading a patch. LLDB differs from the LLVM Developer
Policy in the following respects.

 - **Test infrastructure**: Like LLVM it is  important to submit tests with your
   patches, but note that LLDB uses a different system for tests. Refer to the
   `test documentation <test.html>`_ for more details and the ``lldb/test``
   folder on disk for examples.

 - **Coding Style**: LLDB's code style differs from LLVM's coding style.
   Unfortunately there is no document describing the differences. Please be
   consistent with the existing code.

For anything not explicitly listed here, assume that LLDB follows the LLVM
policy.


Error handling and use of assertions in LLDB
--------------------------------------------

Contrary to Clang, which is typically a short-lived process, LLDB
debuggers stay up and running for a long time, often serving multiple
debug sessions initiated by an IDE. For this reason LLDB code needs to
be extra thoughtful about how to handle errors. Below are a couple
rules of thumb:

* Invalid input.  To deal with invalid input, such as malformed DWARF,
  missing object files, or otherwise inconsistent debug info, LLVM's
  error handling types such as `llvm::Expected<T>
  <https://llvm.org/doxygen/classllvm_1_1Expected.html>`_ or
  `std::optional<T>
  <https://llvm.org/doxygen/classllvm_1_1Optional.html>`_ should be
  used. Functions that may fail should return their result using these
  wrapper types instead of using a bool to indicate success. Returning
  a default value when an error occurred is also discouraged.

* Assertions.  Assertions (from ``assert.h``) should be used liberally
  to assert internal consistency.  Assertions shall **never** be
  used to detect invalid user input, such as malformed DWARF.  An
  assertion should be placed to assert invariants that the developer
  is convinced will always hold, regardless what an end-user does with
  LLDB. Because assertions are not present in release builds, the
  checks in an assertion may be more expensive than otherwise
  permissible. In combination with the LLDB test suite, assertions are
  what allows us to refactor and evolve the LLDB code base.

* Logging. LLDB provides a very rich logging API. When recoverable
  errors cannot reasonably be surfaced to the end user, the error may
  be written to a topical log channel.

* Soft assertions.  LLDB provides ``lldbassert()`` as a soft
  alternative to cover the middle ground of situations that indicate a
  recoverable bug in LLDB.  When asserts are enabled ``lldbassert()``
  behaves like ``assert()``. When asserts are disabled, it will print a
  warning and encourage the user to file a bug report, similar to
  LLVM's crash handler, and then return execution. Use these sparingly
  and only if error handling is not otherwise feasible.  Specifically,
  new code should not be using ``lldbassert()`` and existing
  uses should be replaced by other means of error handling.

* Fatal errors.  Aborting LLDB's process using
  ``llvm::report_fatal_error()`` or ``abort()`` should be avoided at all
  costs.  It's acceptable to use `llvm_unreachable()
  <https://llvm.org/doxygen/Support_2ErrorHandling_8h.html>`_ for
  actually unreachable code such as the default in an otherwise
  exhaustive switch statement.

Overall, please keep in mind that the debugger is often used as a last
resort, and a crash in the debugger is rarely appreciated by the
end-user.
