.. _contributing:

================================
Contributing to the libc Project
================================

LLVM's libc is being developed as part of the LLVM project so contributions
to the libc project should also follow the general LLVM
`contribution guidelines <https://llvm.org/docs/Contributing.html>`_. Below is
a list of open projects that one can start with:

#. **Cleanup code-style** - The libc project follows the general
   `LLVM style <https://llvm.org/docs/CodingStandards.html>`_ but differs in a
   few aspects: We use ``snake_case`` for non-constant variable and function
   names,``CamelCase`` for internal type names (those which are not defined in a
   public header), and ``CAPITALIZED_SNAKE_CASE`` for constants. When we started
   working on the project, we started using the general LLVM style for
   everything. However, for a short period, we switched to the style that is
   currently followed by the `LLD project <https://github.com/llvm/llvm-project/tree/main/lld>`_.
   But, considering that we implement a lot of functions and types whose names
   are prescribed by the standards, we have settled on the style described above.
   However, we have not switched over to this style in all parts of the ``libc``
   directory. So, a simple but mechanical project would be to move the parts
   following the old styles to the new style.

#. **Integrating with the rest of the LLVM project** - There are two parts to
   this project:

   #. One is about adding CMake facilities to optionally link the libc's overlay
      static archive (see :ref:`overlay_mode`) with other LLVM tools/executables.
   #. The other is about putting plumbing in place to release the overlay static
      archive (see :ref:`overlay_mode`) as part of the LLVM binary releases.

#. **Implement Linux syscall wrappers** - A large portion of the POSIX API can
   be implemented as syscall wrappers on Linux. A good number have already been
   implemented but many more are yet to be implemented. So, a project of medium
   complexity would be to implement syscall wrappers which have not yet been
   implemented.

#. **Add a better random number generator** - The current random number
   generator has a very small range. This has to be improved or switched over
   to a fast random number generator with a large range.

#. **Update the clang-tidy lint rules and use them in the build and/or CI** -
   Currently, the :ref:`clang_tidy_checks` have gone stale and are mostly unused
   by the developers and on the CI builders. This project is about updating
   them and reintegrating them back with the build and running them on the
   CI builders.

#. **double and higher precision math functions** - These are under active
   development but you can take a shot at those not yet implemented. See
   :ref:`math` for more information.

#. **Contribute a new OS/Architecture port** - You can contribute a new
   operating system or target architecture port. See :ref:`porting` for more
   information.
