.. _contributing:

================================
Contributing to the libc Project
================================

LLVM-libc is being developed as part of the LLVM project so contributions
to the libc project should also follow the general LLVM
`contribution guidelines <https://llvm.org/docs/Contributing.html>`_. Below is
a list of open projects that one can start with:

#. **Beginner Bugs** - Help us tackle
   `good first issues <https://github.com/llvm/llvm-project/issues?q=is%3Aopen+is%3Aissue+label%3Alibc+label%3A%22good+first+issue%22>`__.
   These bugs have been tagged with the github labels "libc" and "good first
   issue" by the team as potentially easier places to get started.  Please do
   first check if the bug has an assignee; if so please find another unless
   there's been no movement on the issue from the assignee, in which place do
   ask if you can help take over.

#. **Cleanup code-style** - The libc project follows the general
   `LLVM style <https://llvm.org/docs/CodingStandards.html>`_ with specific
   conventions for naming (``snake_case`` for functions, ``CamelCase`` for
   types). See the :ref:`code_style` page for the authoritative reference.
   Mechanical projects to move parts following old styles to the current
   conventions are welcome.

#. **Implement Linux syscall wrappers** - A large portion of the POSIX API can
   be implemented as syscall wrappers on Linux. A good number have already been
   implemented but many more are yet to be implemented. So, a project of medium
   complexity would be to implement syscall wrappers which have not yet been
   implemented.

#. **Update the clang-tidy lint rules and use them in the build and/or CI** -
   The libc project has a set of clang-tidy checks (see :ref:`clang_tidy_checks`)
   but they are not enabled by default. They can be enabled by configuring with
   ``-DLLVM_LIBC_ENABLE_LINTING=ON`` (or by setting ``LLVM_LIBC_CLANG_TIDY``) and
   running the ``libc-lint`` build target. This project is about keeping the
   checks up to date and reintegrating them into the build and CI.

#. **double and higher precision math functions** - These are under active
   development but you can take a shot at those not yet implemented. See
   :ref:`math` for more information.

#. **Contribute a new OS/Architecture port** - You can contribute a new
   operating system or target architecture port. See :ref:`porting` for more
   information.
