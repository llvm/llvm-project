====================
Force-Linker Headers
====================

.. WARNING:: The framework is rapidly evolving.
  The documentation might be out-of-sync with the implementation.
  The purpose of this documentation is to give context for upcoming reviews.

The problem
***********

SSAF uses `llvm::Registry\<\> <https://llvm.org/doxygen/classllvm_1_1Registry.html>`_
for decentralized registration of summary extractors and serialization formats.
Each registration is a file-scope static object whose constructor adds an entry
to the global registry:

.. code-block:: c++

  // In MyExtractor.cpp
  static TUSummaryExtractorRegistry::Add<MyExtractor>
      RegisterExtractor("MyExtractor", "My summary extractor");

When the translation unit containing this static object is compiled into a
**static library** (``.a`` / ``.lib``), the static linker will only pull in
object files that resolve an undefined symbol in the consuming binary.
Because no code ever calls anything in ``MyExtractor.o`` directly, the linker
discards the object file — and the registration never runs.

This is not a problem for **shared libraries** (``.so`` / ``.dylib``), because
the dynamic linker loads the entire shared object and runs all global
constructors unconditionally.

The solution: anchor symbols
****************************

Each registration translation unit defines a ``const volatile int`` **anchor symbol**:

.. code-block:: c++

  // In MyExtractor.cpp - next to the registry Add<> object in the ``clang::ssaf`` namespace
  // NOLINTNEXTLINE(misc-use-internal-linkage)
  const volatile int MyExtractorAnchorSource = 0;

For **in-tree** anchors, add a single ``ANCHOR(...)`` entry to
``BuiltinAnchorSources.def`` (in alphabetical order):

.. code-block:: c++

  // In clang/include/clang/ScalableStaticAnalysisFramework/BuiltinAnchorSources.def
  ANCHOR(JSONFormatAnchorSource)
  ANCHOR(MyExtractorAnchorSource) // <-- Add here, in alphabetical order

``SSAFBuiltinForceLinker.h`` includes this ``.def`` file automatically to
generate the ``extern`` declarations and the ``AnchorSources`` array — there is
no need to edit that header directly.

Any translation unit that ``#include``\s this header now has a reference to
``MyExtractorAnchorSource``, which forces the linker to pull in
``MyExtractor.o`` — and with it, the static ``Add<>`` registration object.

The ``volatile`` qualifier is essential: without it the compiler could
constant-fold the ``0`` and eliminate the reference entirely.

Header hierarchy
================

.. code-block:: text

  SSAFForceLinker.h                   (umbrella — include this in binaries)
  └── SSAFBuiltinForceLinker.h        (upstream built-in anchors only)

- ``clang/include/clang/ScalableStaticAnalysisFramework/SSAFBuiltinForceLinker.h`` — anchors for
  upstream-provided (built-in) extractors and formats (e.g. ``JSONFormat``).
- ``clang/include/clang/ScalableStaticAnalysisFramework/SSAFForceLinker.h`` — umbrella header
  that includes ``SSAFBuiltinForceLinker.h``.  This is the header that
  downstream projects should modify to add their own force-linker includes
  (see :doc:`HowToExtend`).

Include the umbrella header with ``// IWYU pragma: keep`` in any translation
unit that must guarantee all registrations are active — typically the entry
point of a binary that uses ``clangScalableStaticAnalysisFrameworkCore``:

.. code-block:: c++

  // In ExecuteCompilerInvocation.cpp
  #include "clang/ScalableStaticAnalysisFramework/SSAFForceLinker.h" // IWYU pragma: keep

Naming convention
=================

Anchor symbols follow the pattern ``<Component>AnchorSource`` in the ``clang::ssaf`` namespace.
For example:

- ``JSONFormatAnchorSource``
- ``MyExtractorAnchorSource``

All anchor sources are aggregated into a single ``BuiltinAnchorDestination``
lambda in the force-linker header (see ``SSAFBuiltinForceLinker.h``).

Considered alternatives
***********************

``--whole-archive`` / ``-force_load``
=====================================

The linker can be instructed to include *every* object file from a static
library, regardless of whether any symbols are referenced:

.. code-block:: bash

  # GNU ld / lld (Linux, BSD)
  -Wl,--whole-archive -lclangScalableStaticAnalysisFrameworkCore -Wl,--no-whole-archive

  # Apple ld
  -Wl,-force_load,libclangScalableStaticAnalysisFrameworkCore.a

Since CMake 3.24, the ``$<LINK_LIBRARY:WHOLE_ARCHIVE,...>`` generator expression
provides a portable way to do the same:

.. code-block:: cmake

  target_link_libraries(clang PRIVATE
    "$<LINK_LIBRARY:WHOLE_ARCHIVE,clangScalableStaticAnalysisFrameworkCore>")

**Why we did not choose this approach**:

- It is a blunt instrument — *all* object files in the library are pulled in,
  increasing binary size.
- The anchor approach only targets specific object files: only registrations
  whose anchors are referenced in a force-linker header are pulled in.
- ``--whole-archive`` semantics vary across platforms and toolchains, requiring
  platform-specific CMake logic or the relatively new ``WHOLE_ARCHIVE``
  generator expression.

Explicit initialization functions
=================================

An alternative is a central ``initializeSSAFRegistrations()`` function that
explicitly calls into each registration module:

.. code-block:: c++

  void initializeSSAFRegistrations() {
    initializeJSONFormat();
    initializeMyExtractor();
    // ... one entry per registration
  }

**Why we did not choose this approach**:

- It reintroduces a centralized list that must be maintained manually, defeating
  the decoupled-registration benefit of ``llvm::Registry``.
- Adding a new extractor or format requires modifying a central file, which
  increases merge-conflict risk for downstream users.
