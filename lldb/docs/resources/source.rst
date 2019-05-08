Getting the Sources
===================

Refer to the `LLVM Getting Started Guide
<http://llvm.org/docs/GettingStarted.html#getting-started-with-llvm>`_
for general instructions on how to check out source. Note that LLDB
depends on having a working checkout of LLVM and Clang, so the first
step is to download and build as described at the above URL. The same
repository also contains LLDB.

Git browser: https://github.com/llvm/llvm-project/tree/master/lldb

For macOS building from Xcode, simply checkout LLDB and then build
from Xcode. The Xcode project will automatically detect that it is a
fresh checkout, and checkout LLVM and Clang automatically. Unlike
other platforms / build systems, it will use the following directory
structure.

 ::

                  lldb
                  |
                  `-- llvm
                      |
                      +-- tools
                          |
                          `-- clang


So updating your checkout will consist of updating LLDB, LLV<, and
Clang in these locations.  Refer to the `Build Instructions
<build.html>`_ for more detailed instructions on how to build for a
particular platform / build system combination.

Contributing to LLDB
--------------------

Please refer to the `LLVM Developer Policy
<http://llvm.org/docs/DeveloperPolicy.html>`_ for information about
authoring and uploading a patch. LLDB differs from the LLVM Developer
Policy in the following respects.

Test infrastructure. It is still important to submit tests with your
patches, but LLDB uses a different system for tests. Refer to the
`lldb/test` folder on disk for examples of how to write tests.  For
anything not explicitly listed here, assume that LLDB follows the LLVM
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
  <http://llvm.org/doxygen/classllvm_1_1Expected.html>`_ or
  `llvm::Optional<T>
  <http://llvm.org/doxygen/classllvm_1_1Optional.html>`_ should be
  used. Functions that may fail should return their result using these
  wrapper types instead of using a bool to indicate success. Returning
  a default value when an error occurred is also discouraged.

* Assertions.  Assertions (from `assert.h`) should be used liberally
  to assert internal consistency.  Assertions shall **never** be
  used to detect invalid user input, such as malformed DWARF.  An
  assertion should be placed to assert invariants that the developer
  is convinced will always hold, regardless what an end-user does with
  LLDB. Because assertions are not present in release builds, the
  checks in an assertion may be more expensive than otherwise
  permissible. In combination with the LLDB test suite, assertions are
  what allows us to refactor and evolve the LLDB code base.

* Logging. LLDB provides a very rich logging API. When recoverable
  errors cannot reasonably by surfaced to the end user, the error may
  be written to a topical log channel.

* Soft assertions.  LLDB provides `lldb_assert()` as a soft
  alternative to cover the middle ground of situations that indicate a
  recoverable bug in LLDB.  In a Debug configuration `lldb_assert()`
  behaves like `assert()`. In a Release configuration it will print a
  warning and encourage the user to file a bug report, similar to
  LLVM's crash handler, and then return execution. Use these sparingly
  and only if error handling is not otherwise feasible.  Specifically,
  new code should not be using `lldb_assert()` and existing
  uses should be replaced by other means of error handling.

* Fatal errors.  Aborting LLDB's process using
  `llvm::report_fatal_error()` or `abort()` should be avoided at all
  costs.  It's acceptable to use `llvm_unreachable()
  <http://llvm.org/doxygen/Support_2ErrorHandling_8h.html>`_ for
  actually unreachable code such as the default in an otherwise
  exhaustive switch statement.

Overall, please keep in mind that the debugger is often used as a last
resort, and a crash in the debugger is rarely appreciated by the
end-user.
