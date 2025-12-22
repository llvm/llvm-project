==================
Summary Extraction
==================

.. WARNING:: The framework is rapidly evolving.
  The documentation might be out-of-sync of the implementation.
  The purpose of this documentation to give context for upcoming reviews.


The simplest way to think about the lifetime of a summary extraction is by following the handlers of the ``FrontendAction`` implementing it.
There are 3 APIs that are important for us, that are invoked in this order:

  - ``BeingInvocation()``: Checks the command-line arguments related to summary extraction.
  - ``CreateASTConsumer()``: Creates the ASTConsumers for the different summary extractors.
  - ``EndSourceFile()``: Serializes and writes the extracted summaries.

Implementation details
**********************

Global Registries
=================

The framework uses *registries* as an extension point for adding new summary analyses or serialization formats.

A *registry* is basically a global function returning some local static storage housing objects that contain some function pointers.
Think of some cookbook that holds recipes, and the recipe refers to the instructions of how to cook (or *construct*) the *thing*.
Adding to the *registry* (or *cookbook*) can be achieved by creating a translation-unit local static object with a constructor that does this by inserting the given function pointers (*recipe*) to the ``vector/set/map`` of the *registry*.
When the executable starts, it will construct the global objects, thus also applying the side effect of populating the registries with the entries.

**Pros**:

  - Decentralizes the registration. There is not a single place in the source code where we spell out all of the analyses/formats.
  - Plays nicely with downstream extensibility, as downstream users can add their own analyses/formats without touching the source code of the framework; while still benefiting from the upstream-provided analyses/formats.
  - Works with static and dynamic linking. In other words, plugins as shared objects compose naturally.

**Cons**:

  - Registration slows down all ``clang`` users by a tiny amount, even if they don't invoke the summary extraction framework.
  - As the registration is now decoupled, it's now a global program property; and potentially more difficult to reason about.
  - Complicates testing.
  - We have to deal with function pointers, as a layer of indirection, making it harder to debug where the indirect function calls go in an IDE, while statically inspecting the code.

The general idea
----------------

.. code-block:: c++

  //--- SomeRegistry.h
  struct Registrar {
    Registrar(std::string Name, void (*Printer)());
  };
  struct Info {
    void (*Printer)();
    // Place more function pointers if needed.
  };
  std::map<std::string, Info>& getRegistry();

  //--- SomeRegistry.cpp
  std::map<std::string, Info>& getRegistry() {
    static std::map<std::string, Info> Storage;
    return Storage;
  }
  Registrar::Registrar(std::string Name, void (*Printer)()) {
    bool Inserted = getRegistry().try_emplace(std::move(Name), Info{Printer}).second;
    assert(Inserted && "Name was already present in the registry");
    (void)Inserted;
  }

  //--- MyAnalysis.cpp
  extern void MyAnalysisPrinter() {
    printf("MyAnalysisPrinter");
  }
  static Registrar MyAnalysis("awesome-analysis", &MyAnalysisPrinter);

  //--- Framework.cpp
  void print_all() {
    for (const auto &[Name, Entry] : getRegistry()) {
      (*Entry.Printer)(); // Invoke the customized printer.
    }
  }

Details of ``BeingInvocation()``
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
  #. The same goes for each enabled analysis, take the ``EntityID`` to ``TUSummaryData`` mapping and serialize them using the analysis-provided ``Serialize`` function pointer.
