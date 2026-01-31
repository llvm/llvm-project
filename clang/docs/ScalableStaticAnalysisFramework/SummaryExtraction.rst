==================
Summary Extraction
==================

.. WARNING:: The framework is rapidly evolving.
  The documentation might be out-of-sync of the implementation.
  The purpose of this documentation to give context for upcoming reviews.


The simplest way to think about the lifetime of a summary extraction is by following the handlers of the ``FrontendAction`` implementing it.
There are 3 APIs that are important for us, that are invoked in this order:

  - ``BeginInvocation()``: Checks the command-line arguments related to summary extraction.
  - ``CreateASTConsumer()``: Creates the ASTConsumers for the different summary extractors.
  - ``EndSourceFile()``: Serializes and writes the extracted summaries.

Implementation details
**********************

Global Registries
=================

The framework uses `llvm::Registry\<\> <https://llvm.org/doxygen/classllvm_1_1Registry.html>`_
as an extension point for adding new summary analyses or serialization formats.
Each entry in the *registry* holds a name, a description and a pointer to a constructor.

**Pros**:

  - Decentralizes the registration. There is not a single place in the source code where we spell out all of the analyses/formats.
  - Plays nicely with downstream extensibility, as downstream users can add their own analyses/formats without touching the source code of the framework; while still benefiting from the upstream-provided analyses/formats.
  - Works with static and dynamic linking. In other words, plugins as shared objects compose naturally.

**Cons**:

  - Registration slows down all ``clang`` users by a tiny amount, even if they don't invoke the summary extraction framework.
  - As the registration is now decoupled, it's now a global program property; and potentially more difficult to reason about.
  - Complicates testing.

Example for adding a custom summary extraction
----------------------------------------------

.. code-block:: c++

  //--- MyAnalysis.cpp
  class MyAnalysis : public TUSummaryExtractor {
    using TUSummaryExtractor::TUSummaryExtractor;
    // Implementation...
  };

  static TUSummaryExtractorRegistry::Add<MyAnalysis>
    RegisterExtractor("MyAwesomeAnalysis", "The analysis produces some awesome results");

Details of ``BeginInvocation()``
================================

#. Processes the different fields populated from the command line. Ensure that mandatory flags are set, etc.
#. For each requested analysis, check if we have a matching ``TUSummaryExtractorInfo`` in the static registry, and diagnose if not.
#. Parse the format name, and check if we have a matching ``FormatInfo`` in the format registry.
#. Lastly, forward the ``BeginInvocation`` call to the wrapped FrontendAction.


Details of ``CreateASTConsumer()``
==================================

#. Create the wrapped ``FrontendAction`` consumers by calling ``CreateASTConsumer()`` on it.
#. Call ``ssaf::makeTUSummaryExtractor()`` on each requested analysis name.

  #. Look up in the *summary registry* the relevant *Info* object and call the ``Factory`` function pointer to create the relevant ``ASTConsumer``.
  #. Remember, we pass a mutable ``TUSummaryBuilder`` reference to the constructor, so the analysis can create ``EntityID`` objects and map them to ``TUSummaryData`` objects in their implementation. Their custom metadata needs to inherit from ``TUSummaryData`` to achieve this.

#. Lastly, add all of these ``ASTConsumers`` to the ``MultiplexConsumer`` and return that.


Details of ``EndSourceFile()``
==============================

#. Call the virtual ``writeTUSummary()`` on the serialization format, leading to the desired format handler (such as JSON or binary or something custom - provided by a plugin).

  #. Create the directory structure for the enabled analyses.
  #. Serialize ``entities``, ``entity_linkage``, etc. Achieve by calling the matching virtual functions, dispatching to the concrete implementation.
  #. The same goes for each enabled analysis, serialize the ``EntityID`` to ``TUSummaryData`` mapping using the analysis-provided ``Serialize`` function pointer.
