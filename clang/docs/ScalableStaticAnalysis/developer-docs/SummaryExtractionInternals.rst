============================
Summary Extraction Internals
============================

.. WARNING:: The framework is rapidly evolving.
  The documentation might be out-of-sync with the implementation.
  The purpose of this documentation is to give context for upcoming reviews.

When ``--ssaf-tu-summary-file=`` is non-empty, ``CreateFrontendAction()`` (in ``ExecuteCompilerInvocation.cpp``)
wraps the original ``FrontendAction`` inside a ``TUSummaryExtractorFrontendAction``.
This ensures that the summary extraction transparently happens after the original frontend action, which is usually either compilation (``-c``) or just ``-fsyntax-only`` in tests.

Lifetime of a summary extraction
********************************

The ``TUSummaryExtractorFrontendAction`` will try to construct a ``TUSummaryRunner`` ASTConsumer and report an error on failure.
When it succeeds, it will multiplex the handlers of the ASTConsumer to every summary extractor and in the end, serialize and write the results to the desired file.

Implementation details
**********************

Global Registries
=================

The framework uses `llvm::Registry\<\> <https://llvm.org/doxygen/classllvm_1_1Registry.html>`_
as an extension point for adding new summary analyses or serialization formats.
Each entry in the *registry* holds a name, a description and a pointer to a constructor.
Because static linking can discard unreferenced registration objects, the framework
uses :doc:`ForceLinkerHeaders` to ensure they are retained.

For details on how to add new extractors and formats, see :doc:`HowToExtend`.
