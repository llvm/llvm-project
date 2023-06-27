.. title:: clang-tidy - misc-header-include-cycle

misc-header-include-cycle
=========================

Check detects cyclic ``#include`` dependencies between user-defined headers.

.. code-block:: c++

    // Header A.hpp
    #pragma once
    #include "B.hpp"

    // Header B.hpp
    #pragma once
    #include "C.hpp"

    // Header C.hpp
    #pragma once
    #include "A.hpp"

    // Include chain: A->B->C->A

Header files are a crucial part of many C++ programs, as they provide a way to
organize declarations and definitions that are shared across multiple source
files. However, header files can also create problems when they become entangled
in complex dependency cycles. Such cycles can cause issues with compilation
times, unnecessary rebuilds, and make it harder to understand the overall
structure of the code.

To address these issues, this check has been developed. This check is designed
to detect cyclic dependencies between header files, also known as
"include cycles". An include cycle occurs when a header file `A` includes a
header file `B`, and header file `B` (or any later included header file in the
chain) includes back header file `A`, leading to a circular dependency cycle.

This check operates at the preprocessor level and analyzes user-defined headers
and their dependencies. It focuses specifically on detecting include cycles,
and ignores other types or function dependencies. This allows it to provide a
specialized analysis that is focused on identifying and preventing issues
related to header file organization.

The benefits of using this check are numerous. By detecting include cycles early
in the development process, developers can identify and resolve these issues
before they become more difficult and time-consuming to fix. This can lead to
faster compile times, improved code quality, and a more maintainable codebase
overall. Additionally, by ensuring that header files are organized in a way that
avoids cyclic dependencies, developers can make their code easier to understand
and modify over time.

It's worth noting that this tool only analyzes user-defined headers and their
dependencies, excluding system includes such as standard library headers and
third-party library headers. System includes are usually well-designed and free
of include cycles, and ignoring them helps to focus on potential issues within
the project's own codebase. This limitation doesn't diminish the tool's ability
to detect ``#include`` cycles within the analyzed code. As with any tool,
developers should use their judgment when evaluating the warnings produced by
the check and be prepared to make exceptions or modifications to their code as
needed.

Options
-------

.. option:: IgnoredFilesList

    Provides a way to exclude specific files/headers from the warnings raised by
    a check. This can be achieved by specifying a semicolon-separated list of
    regular expressions or filenames. This option can be used as an alternative
    to ``//NOLINT`` when using it is not possible.
    The default value of this option is an empty string, indicating that no
    files are ignored by default.
