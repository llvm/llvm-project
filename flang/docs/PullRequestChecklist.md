<!--===- docs/PullRequestChecklist.md 

   Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
   See https://llvm.org/LICENSE.txt for license information.
   SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

-->

# Pull request checklist
Please review the following items before submitting a pull request.  This list
can also be used when reviewing pull requests.
*  Verify that new files have a license with correct file name.
*  Run `git diff` on all modified files to look for spurious changes such as
   `#include <iostream>`.
*  If you added code that causes the compiler to emit a new error message, make
   sure that you also added a test that causes that error message to appear
   and verifies its correctness.
*  Annotate the code and tests with appropriate references to constraint and
   requirement numbers from the Fortran standard.  Do not include the text of
   the constraint or requirement, just its number.
*  Alphabetize arbitrary lists of names.
*  Check dereferences of pointers and optionals where necessary.
*  Ensure that the scopes of all functions and variables are as local as
   possible.
*  Try to make all functions fit on a screen (40 lines).
*  Build and test with both GNU and clang compilers.
*  When submitting an update to a pull request, review previous pull request
   comments and make sure that you've actually made all of the changes that
   were requested.

## Follow the style guide
The following items are taken from the [C++ style guide](C++style.md).  But
even though I've read the style guide, they regularly trip me up.
*  Run clang-format using the git-clang-format script from LLVM HEAD.
*  Make sure that all source lines have 80 or fewer characters.  Note that
   clang-format will do this for most code.  But you may need to break up long
   strings.
*  Review declarations for proper use of `constexpr` and `const`.
*  Follow the C++ [naming guidelines](C++style.html#naming)
*  Ensure that the names evoke their purpose and are consistent with existing code.
*  Used braced initializers.
*  Review pointer and reference types to make sure that you're using them
   appropriately.  Note that the [C++ style guide](C++style.md) contains a
   section that describes all of the pointer types along with their
   characteristics.
*  Declare non-member functions ```static``` when possible.  Prefer
   ```static``` functions over functions in anonymous namespaces.
