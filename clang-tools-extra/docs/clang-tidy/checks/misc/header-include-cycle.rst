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

Header files are a crucial part of many C++ programs as they provide a way to
organize declarations and definitions shared across multiple source files.
However, header files can also create problems when they become entangled
in complex dependency cycles. Such cycles can cause issues with compilation
times, unnecessary rebuilds, and make it harder to understand the overall
structure of the code.

To address these issues, a check has been developed to detect cyclic
dependencies between header files, also known as "include cycles".
An include cycle occurs when a header file `A` includes header file `B`,
and `B` (or any subsequent included header file) includes back header file `A`,
resulting in a circular dependency cycle.

This check operates at the preprocessor level and specifically analyzes
user-defined headers and their dependencies. It focuses solely on detecting
include cycles while disregarding other types or function dependencies.
This specialized analysis helps identify and prevent issues related to header
file organization.

By detecting include cycles early in the development process, developers can
identify and resolve these issues before they become more difficult and
time-consuming to fix. This can lead to faster compile times, improved code
quality, and a more maintainable codebase overall. Additionally, by ensuring
that header files are organized in a way that avoids cyclic dependencies,
developers can make their code easier to understand and modify over time.

It's worth noting that only user-defined headers their dependencies are analyzed,
System includes such as standard library headers and third-party library headers
are excluded. System includes are usually well-designed and free of include
cycles, and ignoring them helps to focus on potential issues within the
project's own codebase. This limitation doesn't diminish the ability to detect
``#include`` cycles within the analyzed code.

Developers should carefully review any warnings or feedback provided by this
solution. While the analysis aims to identify and prevent include cycles, there
may be situations where exceptions or modifications are necessary. It's
important to exercise judgment and consider the specific context of the codebase
when making adjustments.

Options
-------

.. option:: IgnoredFilesList

    Provides a way to exclude specific files/headers from the warnings raised by
    a check. This can be achieved by specifying a semicolon-separated list of
    regular expressions or filenames. This option can be used as an alternative
    to ``//NOLINT`` when using it is not possible.
    The default value of this option is an empty string, indicating that no
    files are ignored by default.
