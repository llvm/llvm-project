================
Getting Involved
================

:program:`clang-tidy` has several own checks and can run Clang static analyzer
checks, but its power is in the ability to easily write custom checks.

Checks are organized in modules, which can be linked into :program:`clang-tidy`
with minimal or no code changes in :program:`clang-tidy`.

Checks can plug into the analysis on the preprocessor level using `PPCallbacks`_
or on the AST level using `AST Matchers`_. When an error is found, checks can
report them in a way similar to how Clang diagnostics work. A fix-it hint can be
attached to a diagnostic message.

The interface provided by :program:`clang-tidy` makes it easy to write useful
and precise checks in just a few lines of code. If you have an idea for a good
check, the rest of this document explains how to do this.

There are a few tools particularly useful when developing clang-tidy checks:
  * ``add_new_check.py`` is a script to automate the process of adding a new
    check, it will create the check, update the CMake file and create a test;
  * ``rename_check.py`` does what the script name suggests, renames an existing
    check;
  * :program:`pp-trace` logs method calls on `PPCallbacks` for a source file
    and is invaluable in understanding the preprocessor mechanism;
  * :program:`clang-query` is invaluable for interactive prototyping of AST
    matchers and exploration of the Clang AST;
  * `clang-check`_ with the ``-ast-dump`` (and optionally ``-ast-dump-filter``)
    provides a convenient way to dump AST of a C++ program.

If CMake is configured with ``CLANG_TIDY_ENABLE_STATIC_ANALYZER=NO``,
:program:`clang-tidy` will not be built with support for the
``clang-analyzer-*`` checks or the ``mpi-*`` checks.


.. _AST Matchers: https://clang.llvm.org/docs/LibASTMatchers.html
.. _PPCallbacks: https://clang.llvm.org/doxygen/classclang_1_1PPCallbacks.html
.. _clang-check: https://clang.llvm.org/docs/ClangCheck.html


Choosing the Right Place for your Check
---------------------------------------

If you have an idea of a check, you should decide whether it should be
implemented as a:

+ *Clang diagnostic*: if the check is generic enough, targets code patterns that
  most probably are bugs (rather than style or readability issues), can be
  implemented effectively and with extremely low false positive rate, it may
  make a good Clang diagnostic.

+ *Clang static analyzer check*: if the check requires some sort of control flow
  analysis, it should probably be implemented as a static analyzer check.

+ *clang-tidy check* is a good choice for linter-style checks, checks that are
  related to a certain coding style, checks that address code readability, etc.


Preparing your Workspace
------------------------

If you are new to LLVM development, you should read the `Getting Started with
the LLVM System`_, `Using Clang Tools`_ and `How To Setup Clang Tooling For
LLVM`_ documents to check out and build LLVM, Clang and Clang Extra Tools with
CMake.

Once you are done, change to the ``llvm/clang-tools-extra`` directory, and
let's start!

.. _Getting Started with the LLVM System: https://llvm.org/docs/GettingStarted.html
.. _Using Clang Tools: https://clang.llvm.org/docs/ClangTools.html
.. _How To Setup Clang Tooling For LLVM: https://clang.llvm.org/docs/HowToSetupToolingForLLVM.html

When you `configure the CMake build <https://llvm.org/docs/GettingStarted.html#local-llvm-configuration>`_,
make sure that you enable the ``clang`` and ``clang-tools-extra`` projects to
build :program:`clang-tidy`.
Because your new check will have associated documentation, you will also want to install
`Sphinx <https://www.sphinx-doc.org/en/master/>`_ and enable it in the CMake configuration.
To save build time of the core Clang libraries you may want to only enable the ``X86``
target in the CMake configuration.


The Directory Structure
-----------------------

:program:`clang-tidy` source code resides in the
``llvm/clang-tools-extra`` directory and is structured as follows:

::

  clang-tidy/                       # Clang-tidy core.
  |-- ClangTidy.h                   # Interfaces for users.
  |-- ClangTidyCheck.h              # Interfaces for checks.
  |-- ClangTidyModule.h             # Interface for clang-tidy modules.
  |-- ClangTidyModuleRegistry.h     # Interface for registering of modules.
     ...
  |-- google/                       # Google clang-tidy module.
  |-+
    |-- GoogleTidyModule.cpp
    |-- GoogleTidyModule.h
          ...
  |-- llvm/                         # LLVM clang-tidy module.
  |-+
    |-- LLVMTidyModule.cpp
    |-- LLVMTidyModule.h
          ...
  |-- objc/                         # Objective-C clang-tidy module.
  |-+
    |-- ObjCTidyModule.cpp
    |-- ObjCTidyModule.h
          ...
  |-- tool/                         # Sources of the clang-tidy binary.
          ...
  test/clang-tidy/                  # Integration tests.
      ...
  unittests/clang-tidy/             # Unit tests.
  |-- ClangTidyTest.h
  |-- GoogleModuleTest.cpp
  |-- LLVMModuleTest.cpp
  |-- ObjCModuleTest.cpp
      ...


Writing a clang-tidy Check
--------------------------

So you have an idea of a useful check for :program:`clang-tidy`.

First, if you're not familiar with LLVM development, read through the `Getting
Started with LLVM`_ document for instructions on setting up your workflow and
the `LLVM Coding Standards`_ document to familiarize yourself with the coding
style used in the project. For code reviews we mostly use `LLVM Phabricator`_.

.. _Getting Started with LLVM: https://llvm.org/docs/GettingStarted.html
.. _LLVM Coding Standards: https://llvm.org/docs/CodingStandards.html
.. _LLVM Phabricator: https://llvm.org/docs/Phabricator.html

Next, you need to decide which module the check belongs to. Modules
are located in subdirectories of `clang-tidy/
<https://github.com/llvm/llvm-project/tree/main/clang-tools-extra/clang-tidy/>`_
and contain checks targeting a certain aspect of code quality (performance,
readability, etc.), certain coding style or standard (Google, LLVM, CERT, etc.)
or a widely used API (e.g. MPI). Their names are the same as the user-facing
check group names described :ref:`above <checks-groups-table>`.

After choosing the module and the name for the check, run the
``clang-tidy/add_new_check.py`` script to create the skeleton of the check and
plug it to :program:`clang-tidy`. It's the recommended way of adding new checks.

If we want to create a `readability-awesome-function-names`, we would run:

.. code-block:: console

  $ clang-tidy/add_new_check.py readability awesome-function-names


The ``add_new_check.py`` script will:
  * create the class for your check inside the specified module's directory and
    register it in the module and in the build system;
  * create a lit test file in the ``test/clang-tidy/`` directory;
  * create a documentation file and include it into the
    ``docs/clang-tidy/checks/list.rst``.

Let's see in more detail at the check class definition:

.. code-block:: c++

  ...

  #include "../ClangTidyCheck.h"

  namespace clang {
  namespace tidy {
  namespace readability {

  ...
  class AwesomeFunctionNamesCheck : public ClangTidyCheck {
  public:
    AwesomeFunctionNamesCheck(StringRef Name, ClangTidyContext *Context)
        : ClangTidyCheck(Name, Context) {}
    void registerMatchers(ast_matchers::MatchFinder *Finder) override;
    void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  };

  } // namespace readability
  } // namespace tidy
  } // namespace clang

  ...

Constructor of the check receives the ``Name`` and ``Context`` parameters, and
must forward them to the ``ClangTidyCheck`` constructor.

In our case the check needs to operate on the AST level and it overrides the
``registerMatchers`` and ``check`` methods. If we wanted to analyze code on the
preprocessor level, we'd need instead to override the ``registerPPCallbacks``
method.

In the ``registerMatchers`` method we create an AST Matcher (see `AST Matchers`_
for more information) that will find the pattern in the AST that we want to
inspect. The results of the matching are passed to the ``check`` method, which
can further inspect them and report diagnostics.

.. code-block:: c++

  using namespace ast_matchers;

  void AwesomeFunctionNamesCheck::registerMatchers(MatchFinder *Finder) {
    Finder->addMatcher(functionDecl().bind("x"), this);
  }

  void AwesomeFunctionNamesCheck::check(const MatchFinder::MatchResult &Result) {
    const auto *MatchedDecl = Result.Nodes.getNodeAs<FunctionDecl>("x");
    if (!MatchedDecl->getIdentifier() || MatchedDecl->getName().startswith("awesome_"))
      return;
    diag(MatchedDecl->getLocation(), "function %0 is insufficiently awesome")
        << MatchedDecl
        << FixItHint::CreateInsertion(MatchedDecl->getLocation(), "awesome_");
  }

(If you want to see an example of a useful check, look at
`clang-tidy/google/ExplicitConstructorCheck.h
<https://github.com/llvm/llvm-project/blob/main/clang-tools-extra/clang-tidy/google/ExplicitConstructorCheck.h>`_
and `clang-tidy/google/ExplicitConstructorCheck.cpp
<https://reviews.llvm.org/diffusion/L/browse/clang-tools-extra/trunk/clang-tidy/google/ExplicitConstructorCheck.cpp>`_).

If you need to interact with macros or preprocessor directives, you will want to
override the method ``registerPPCallbacks``.  The ``add_new_check.py`` script
does not generate an override for this method in the starting point for your
new check.

If your check applies only under a specific set of language options, be sure
to override the method ``isLanguageVersionSupported`` to reflect that.

Check development tips
----------------------

Writing your first check can be a daunting task, particularly if you are unfamiliar
with the LLVM and Clang code bases.  Here are some suggestions for orienting yourself
in the codebase and working on your check incrementally.

Guide to useful documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Many of the support classes created for LLVM are used by Clang, such as `StringRef
<https://llvm.org/docs/ProgrammersManual.html#the-stringref-class>`_
and `SmallVector <https://llvm.org/docs/ProgrammersManual.html#llvm-adt-smallvector-h>`_.
These and other commonly used classes are described in the `Important and useful LLVM APIs
<https://llvm.org/docs/ProgrammersManual.html#important-and-useful-llvm-apis>`_ and
`Picking the Right Data Structure for the Task
<https://llvm.org/docs/ProgrammersManual.html#picking-the-right-data-structure-for-a-task>`_
sections of the `LLVM Programmer's Manual
<https://llvm.org/docs/ProgrammersManual.html>`_.  You don't need to memorize all the
details of these classes; the generated `doxygen documentation <https://llvm.org/doxygen/>`_
has everything if you need it.  In the header `LLVM/ADT/STLExtras.h
<https://llvm.org/doxygen/STLExtras_8h.html>`_ you'll find useful versions of the STL
algorithms that operate on LLVM containers, such as `llvm::all_of
<https://llvm.org/doxygen/STLExtras_8h.html#func-members>`_.

Clang is implemented on top of LLVM and introduces its own set of classes that you
will interact with while writing your check.  When a check issues diagnostics and
fix-its, these are associated with locations in the source code.  Source code locations,
source files, ranges of source locations and the `SourceManager
<https://clang.llvm.org/doxygen/classclang_1_1SourceManager.html>`_ class provide
the mechanisms for describing such locations.  These and
other topics are described in the `"Clang" CFE Internals Manual
<https://clang.llvm.org/docs/InternalsManual.html>`_.  Whereas the doxygen generated
documentation serves as a reference to the internals of Clang, this document serves
as a guide to other developers.  Topics in that manual of interest to a check developer
are:

- `The Clang "Basic" Library
  <https://clang.llvm.org/docs/InternalsManual.html#the-clang-basic-library>`_ for
  information about diagnostics, fix-it hints and source locations.
- `The Lexer and Preprocessor Library
  <https://clang.llvm.org/docs/InternalsManual.html#the-lexer-and-preprocessor-library>`_
  for information about tokens, lexing (transforming characters into tokens) and the
  preprocessor.
- `The AST Library
  <https://clang.llvm.org/docs/InternalsManual.html#the-lexer-and-preprocessor-library>`_
  for information about how C++ source statements are represented as an abstract syntax
  tree (AST).

Most checks will interact with C++ source code via the AST.  Some checks will interact
with the preprocessor.  The input source file is lexed and preprocessed and then parsed
into the AST.  Once the AST is fully constructed, the check is run by applying the check's
registered AST matchers against the AST and invoking the check with the set of matched
nodes from the AST.  Monitoring the actions of the preprocessor is detached from the
AST construction, but a check can collect information during preprocessing for later
use by the check when nodes are matched by the AST.

Every syntactic (and sometimes semantic) element of the C++ source code is represented by
different classes in the AST.  You select the portions of the AST you're interested in
by composing AST matcher functions.  You will want to study carefully the `AST Matcher
Reference <https://clang.llvm.org/docs/LibASTMatchersReference.html>`_ to understand
the relationship between the different matcher functions.

Using the Transformer library
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Transformer library allows you to write a check that transforms source code by
expressing the transformation as a ``RewriteRule``.  The Transformer library provides
functions for composing edits to source code to create rewrite rules.  Unless you need
to perform low-level source location manipulation, you may want to consider writing your
check with the Transformer library.  The `Clang Transformer Tutorial
<https://clang.llvm.org/docs/ClangTransformerTutorial.html>`_ describes the Transformer
library in detail.

To use the Transformer library, make the following changes to the code generated by
the ``add_new_check.py`` script:

- Include ``../utils/TransformerClangTidyCheck.h`` instead of ``../ClangTidyCheck.h``
- Change the base class of your check from ``ClangTidyCheck`` to ``TransformerClangTidyCheck``
- Delete the override of the ``registerMatchers`` and ``check`` methods in your check class.
- Write a function that creates the ``RewriteRule`` for your check.
- Call the function in your check's constructor to pass the rewrite rule to
  ``TransformerClangTidyCheck``'s constructor.

Developing your check incrementally
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The best way to develop your check is to start with the simple test cases and increase
complexity incrementally.  The test file created by the ``add_new_check.py`` script is
a starting point for your test cases.  A rough outline of the process looks like this:

- Write a test case for your check.
- Prototype matchers on the test file using :program:`clang-query`.
- Capture the working matchers in the ``registerMatchers`` method.
- Issue the necessary diagnostics and fix-its in the ``check`` method.
- Add the necessary ``CHECK-MESSAGES`` and ``CHECK-FIXES`` annotations to your
  test case to validate the diagnostics and fix-its.
- Build the target ``check-clang-tool`` to confirm the test passes.
- Repeat the process until all aspects of your check are covered by tests.

The quickest way to prototype your matcher is to use :program:`clang-query` to
interactively build up your matcher.  For complicated matchers, build up a matching
expression incrementally and use :program:`clang-query`'s ``let`` command to save named
matching expressions to simplify your matcher.  Just like breaking up a huge function
into smaller chunks with intention-revealing names can help you understand a complex
algorithm, breaking up a matcher into smaller matchers with intention-revealing names
can help you understand a complicated matcher.  Once you have a working matcher, the
C++ API will be virtually identical to your interactively constructed matcher.  You can
use local variables to preserve your intention-revealing names that you applied to
nested matchers.

Creating private matchers
^^^^^^^^^^^^^^^^^^^^^^^^^

Sometimes you want to match a specific aspect of the AST that isn't provided by the
existing AST matchers.  You can create your own private matcher using the same
infrastructure as the public matchers.  A private matcher can simplify the processing
in your ``check`` method by eliminating complex hand-crafted AST traversal of the
matched nodes.  Using the private matcher allows you to select the desired portions
of the AST directly in the matcher and refer to it by a bound name in the ``check``
method.

Unit testing helper code
^^^^^^^^^^^^^^^^^^^^^^^^

Private custom matchers are a good example of auxiliary support code for your check
that can be tested with a unit test.  It will be easier to test your matchers or
other support classes by writing a unit test than by writing a ``FileCheck`` integration
test.  The ``ASTMatchersTests`` target contains unit tests for the public AST matcher
classes and is a good source of testing idioms for matchers.

You can build the Clang-tidy unit tests by building the ``ClangTidyTests`` target.
Test targets in LLVM and Clang are excluded from the "build all" style action of
IDE-based CMake generators, so you need to explicitly build the target for the unit
tests to be built.

Making your check robust
^^^^^^^^^^^^^^^^^^^^^^^^

Once you've covered your check with the basic "happy path" scenarios, you'll want to
torture your check with as many edge cases as you can cover in order to ensure your
check is robust.  Running your check on a large code base, such as Clang/LLVM, is a
good way to catch things you forgot to account for in your matchers.  However, the
LLVM code base may be insufficient for testing purposes as it was developed against a
particular set of coding styles and quality measures.  The larger the corpus of code
the check is tested against, the higher confidence the community will have in the
check's efficacy and false positive rate.

Some suggestions to ensure your check is robust:

- Create header files that contain code matched by your check.
- Validate that fix-its are properly applied to test header files with
  :program:`clang-tidy`.  You will need to perform this test manually until
  automated support for checking messages and fix-its is added to the
  ``check_clang_tidy.py`` script.
- Define macros that contain code matched by your check.
- Define template classes that contain code matched by your check.
- Define template specializations that contain code matched by your check.
- Test your check under both Windows and Linux environments.
- Watch out for high false positive rates.  Ideally, a check would have no false
  positives, but given that matching against an AST is not control- or data flow-
  sensitive, a number of false positives are expected.  The higher the false
  positive rate, the less likely the check will be adopted in practice.
  Mechanisms should be put in place to help the user manage false positives.
- There are two primary mechanisms for managing false positives: supporting a
  code pattern which allows the programmer to silence the diagnostic in an ad
  hoc manner and check configuration options to control the behavior of the check.
- Consider supporting a code pattern to allow the programmer to silence the
  diagnostic whenever such a code pattern can clearly express the programmer's
  intent.  For example, allowing an explicit cast to ``void`` to silence an
  unused variable diagnostic.
- Consider adding check configuration options to allow the user to opt into
  more aggressive checking behavior without burdening users for the common
  high-confidence cases.

Documenting your check
^^^^^^^^^^^^^^^^^^^^^^

The ``add_new_check.py`` script creates entries in the
`release notes <https://clang.llvm.org/extra/ReleaseNotes.html>`_, the list of
checks and a new file for the check documentation itself.  It is recommended that you
have a concise summation of what your check does in a single sentence that is repeated
in the release notes, as the first sentence in the doxygen comments in the header file
for your check class and as the first sentence of the check documentation.  Avoid the
phrase "this check" in your check summation and check documentation.

If your check relates to a published coding guideline (C++ Core Guidelines, MISRA, etc.)
or style guide, provide links to the relevant guideline or style guide sections in your
check documentation.

Provide enough examples of the diagnostics and fix-its provided by the check so that a
user can easily understand what will happen to their code when the check is run.
If there are exceptions or limitations to your check, document them thoroughly.  This
will help users understand the scope of the diagnostics and fix-its provided by the check.

Building the target ``docs-clang-tools-html`` will run the Sphinx documentation generator
and create documentation HTML files in the tools/clang/tools/extra/docs/html directory in
your build tree.  Make sure that your check is correctly shown in the release notes and the
list of checks.  Make sure that the formatting and structure of your check's documentation
looks correct.


Registering your Check
----------------------

(The ``add_new_check.py`` script takes care of registering the check in an existing
module. If you want to create a new module or know the details, read on.)

The check should be registered in the corresponding module with a distinct name:

.. code-block:: c++

  class MyModule : public ClangTidyModule {
   public:
    void addCheckFactories(ClangTidyCheckFactories &CheckFactories) override {
      CheckFactories.registerCheck<ExplicitConstructorCheck>(
          "my-explicit-constructor");
    }
  };

Now we need to register the module in the ``ClangTidyModuleRegistry`` using a
statically initialized variable:

.. code-block:: c++

  static ClangTidyModuleRegistry::Add<MyModule> X("my-module",
                                                  "Adds my lint checks.");


When using LLVM build system, we need to use the following hack to ensure the
module is linked into the :program:`clang-tidy` binary:

Add this near the ``ClangTidyModuleRegistry::Add<MyModule>`` variable:

.. code-block:: c++

  // This anchor is used to force the linker to link in the generated object file
  // and thus register the MyModule.
  volatile int MyModuleAnchorSource = 0;

And this to the main translation unit of the :program:`clang-tidy` binary (or
the binary you link the ``clang-tidy`` library in)
``clang-tidy/tool/ClangTidyMain.cpp``:

.. code-block:: c++

  // This anchor is used to force the linker to link the MyModule.
  extern volatile int MyModuleAnchorSource;
  static int MyModuleAnchorDestination = MyModuleAnchorSource;


Configuring Checks
------------------

If a check needs configuration options, it can access check-specific options
using the ``Options.get<Type>("SomeOption", DefaultValue)`` call in the check
constructor. In this case the check should also override the
``ClangTidyCheck::storeOptions`` method to make the options provided by the
check discoverable. This method lets :program:`clang-tidy` know which options
the check implements and what the current values are (e.g. for the
``-dump-config`` command line option).

.. code-block:: c++

  class MyCheck : public ClangTidyCheck {
    const unsigned SomeOption1;
    const std::string SomeOption2;

  public:
    MyCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context),
        SomeOption(Options.get("SomeOption1", -1U)),
        SomeOption(Options.get("SomeOption2", "some default")) {}

    void storeOptions(ClangTidyOptions::OptionMap &Opts) override {
      Options.store(Opts, "SomeOption1", SomeOption1);
      Options.store(Opts, "SomeOption2", SomeOption2);
    }
    ...

Assuming the check is registered with the name "my-check", the option can then
be set in a ``.clang-tidy`` file in the following way:

.. code-block:: yaml

  CheckOptions:
    my-check.SomeOption1: 123
    my-check.SomeOption2: 'some other value'

If you need to specify check options on a command line, you can use the inline
YAML format:

.. code-block:: console

  $ clang-tidy -config="{CheckOptions: {a: b, x: y}}" ...


Testing Checks
--------------

To run tests for :program:`clang-tidy`, build the ``check-clang-tools`` target.
For instance, if you configured your CMake build with the ninja project generator,
use the command:

.. code-block:: console

  $ ninja check-clang-tools

:program:`clang-tidy` checks can be tested using either unit tests or
`lit`_ tests. Unit tests may be more convenient to test complex replacements
with strict checks. `Lit`_ tests allow using partial text matching and regular
expressions which makes them more suitable for writing compact tests for
diagnostic messages.

The ``check_clang_tidy.py`` script provides an easy way to test both
diagnostic messages and fix-its. It filters out ``CHECK`` lines from the test
file, runs :program:`clang-tidy` and verifies messages and fixes with two
separate `FileCheck`_ invocations: once with FileCheck's directive
prefix set to ``CHECK-MESSAGES``, validating the diagnostic messages,
and once with the directive prefix set to ``CHECK-FIXES``, running
against the fixed code (i.e., the code after generated fix-its are
applied). In particular, ``CHECK-FIXES:`` can be used to check
that code was not modified by fix-its, by checking that it is present
unchanged in the fixed code. The full set of `FileCheck`_ directives
is available (e.g., ``CHECK-MESSAGES-SAME:``, ``CHECK-MESSAGES-NOT:``), though
typically the basic ``CHECK`` forms (``CHECK-MESSAGES`` and ``CHECK-FIXES``)
are sufficient for clang-tidy tests. Note that the `FileCheck`_
documentation mostly assumes the default prefix (``CHECK``), and hence
describes the directive as ``CHECK:``, ``CHECK-SAME:``, ``CHECK-NOT:``, etc.
Replace ``CHECK`` by either ``CHECK-FIXES`` or ``CHECK-MESSAGES`` for
clang-tidy tests.

An additional check enabled by ``check_clang_tidy.py`` ensures that
if `CHECK-MESSAGES:` is used in a file then every warning or error
must have an associated CHECK in that file. Or, you can use ``CHECK-NOTES:``
instead, if you want to **also** ensure that all the notes are checked.

To use the ``check_clang_tidy.py`` script, put a .cpp file with the
appropriate ``RUN`` line in the ``test/clang-tidy`` directory. Use
``CHECK-MESSAGES:`` and ``CHECK-FIXES:`` lines to write checks against
diagnostic messages and fixed code.

It's advised to make the checks as specific as possible to avoid checks matching
to incorrect parts of the input. Use ``[[@LINE+X]]``/``[[@LINE-X]]``
substitutions and distinct function and variable names in the test code.

Here's an example of a test using the ``check_clang_tidy.py`` script (the full
source code is at `test/clang-tidy/checkers/google/readability-casting.cpp`_):

.. code-block:: c++

  // RUN: %check_clang_tidy %s google-readability-casting %t

  void f(int a) {
    int b = (int)a;
    // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: redundant cast to the same type [google-readability-casting]
    // CHECK-FIXES: int b = a;
  }

To check more than one scenario in the same test file use
``-check-suffix=SUFFIX-NAME`` on ``check_clang_tidy.py`` command line or
``-check-suffixes=SUFFIX-NAME-1,SUFFIX-NAME-2,...``.
With ``-check-suffix[es]=SUFFIX-NAME`` you need to replace your ``CHECK-*``
directives with ``CHECK-MESSAGES-SUFFIX-NAME`` and ``CHECK-FIXES-SUFFIX-NAME``.

Here's an example:

.. code-block:: c++

   // RUN: %check_clang_tidy -check-suffix=USING-A %s misc-unused-using-decls %t -- -- -DUSING_A
   // RUN: %check_clang_tidy -check-suffix=USING-B %s misc-unused-using-decls %t -- -- -DUSING_B
   // RUN: %check_clang_tidy %s misc-unused-using-decls %t
   ...
   // CHECK-MESSAGES-USING-A: :[[@LINE-8]]:10: warning: using decl 'A' {{.*}}
   // CHECK-MESSAGES-USING-B: :[[@LINE-7]]:10: warning: using decl 'B' {{.*}}
   // CHECK-MESSAGES: :[[@LINE-6]]:10: warning: using decl 'C' {{.*}}
   // CHECK-FIXES-USING-A-NOT: using a::A;$
   // CHECK-FIXES-USING-B-NOT: using a::B;$
   // CHECK-FIXES-NOT: using a::C;$

There are many dark corners in the C++ language, and it may be difficult to make
your check work perfectly in all cases, especially if it issues fix-it hints. The
most frequent pitfalls are macros and templates:

1. code written in a macro body/template definition may have a different meaning
   depending on the macro expansion/template instantiation;
2. multiple macro expansions/template instantiations may result in the same code
   being inspected by the check multiple times (possibly, with different
   meanings, see 1), and the same warning (or a slightly different one) may be
   issued by the check multiple times; :program:`clang-tidy` will deduplicate
   _identical_ warnings, but if the warnings are slightly different, all of them
   will be shown to the user (and used for applying fixes, if any);
3. making replacements to a macro body/template definition may be fine for some
   macro expansions/template instantiations, but easily break some other
   expansions/instantiations.

If you need multiple files to exercise all the aspects of your check, it is
recommended you place them in a subdirectory named for the check under the ``Inputs``
directory for the module containing your check.  This keeps the test directory from
getting cluttered.

If you need to validate how your check interacts with system header files, a set
of simulated system header files is located in the ``checkers/Inputs/Headers``
directory.  The path to this directory is available in a lit test with the variable
``%clang_tidy_headers``.

.. _lit: https://llvm.org/docs/CommandGuide/lit.html
.. _FileCheck: https://llvm.org/docs/CommandGuide/FileCheck.html
.. _test/clang-tidy/checkers/google/readability-casting.cpp: https://github.com/llvm/llvm-project/blob/main/clang-tools-extra/test/clang-tidy/checkers/google/readability-casting.cpp

Out-of-tree check plugins
-------------------------

Developing an out-of-tree check as a plugin largely follows the steps
outlined above. The plugin is a shared library whose code lives outside
the clang-tidy build system. Build and link this shared library against
LLVM as done for other kinds of Clang plugins.

The plugin can be loaded by passing `-load` to `clang-tidy` in addition to the
names of the checks to enable.

.. code-block:: console

  $ clang-tidy --checks=-*,my-explicit-constructor -list-checks -load myplugin.so

There is no expectations regarding ABI and API stability, so the plugin must be
compiled against the version of clang-tidy that will be loading the plugin.

The plugins can use threads, TLS, or any other facilities available to in-tree
code which is accessible from the external headers.

Running clang-tidy on LLVM
--------------------------

To test a check it's best to try it out on a larger code base. LLVM and Clang
are the natural targets as you already have the source code around. The most
convenient way to run :program:`clang-tidy` is with a compile command database;
CMake can automatically generate one, for a description of how to enable it see
`How To Setup Clang Tooling For LLVM`_. Once ``compile_commands.json`` is in
place and a working version of :program:`clang-tidy` is in ``PATH`` the entire
code base can be analyzed with ``clang-tidy/tool/run-clang-tidy.py``. The script
executes :program:`clang-tidy` with the default set of checks on every
translation unit in the compile command database and displays the resulting
warnings and errors. The script provides multiple configuration flags.

.. _How To Setup Clang Tooling For LLVM: https://clang.llvm.org/docs/HowToSetupToolingForLLVM.html


* The default set of checks can be overridden using the ``-checks`` argument,
  taking the identical format as :program:`clang-tidy` does. For example
  ``-checks=-*,modernize-use-override`` will run the ``modernize-use-override``
  check only.

* To restrict the files examined you can provide one or more regex arguments
  that the file names are matched against.
  ``run-clang-tidy.py clang-tidy/.*Check\.cpp`` will only analyze clang-tidy
  checks. It may also be necessary to restrict the header files that warnings
  are displayed from using the ``-header-filter`` flag. It has the same behavior
  as the corresponding :program:`clang-tidy` flag.

* To apply suggested fixes ``-fix`` can be passed as an argument. This gathers
  all changes in a temporary directory and applies them. Passing ``-format``
  will run clang-format over changed lines.


On checks profiling
-------------------

:program:`clang-tidy` can collect per-check profiling info, and output it
for each processed source file (translation unit).

To enable profiling info collection, use the ``-enable-check-profile`` argument.
The timings will be output to ``stderr`` as a table. Example output:

.. code-block:: console

  $ clang-tidy -enable-check-profile -checks=-*,readability-function-size source.cpp
  ===-------------------------------------------------------------------------===
                            clang-tidy checks profiling
  ===-------------------------------------------------------------------------===
    Total Execution Time: 1.0282 seconds (1.0258 wall clock)

     ---User Time---   --System Time--   --User+System--   ---Wall Time---  --- Name ---
     0.9136 (100.0%)   0.1146 (100.0%)   1.0282 (100.0%)   1.0258 (100.0%)  readability-function-size
     0.9136 (100.0%)   0.1146 (100.0%)   1.0282 (100.0%)   1.0258 (100.0%)  Total

It can also store that data as JSON files for further processing. Example output:

.. code-block:: console

  $ clang-tidy -enable-check-profile -store-check-profile=. -checks=-*,readability-function-size source.cpp
  $ # Note that there won't be timings table printed to the console.
  $ ls /tmp/out/
  20180516161318717446360-source.cpp.json
  $ cat 20180516161318717446360-source.cpp.json
  {
  "file": "/path/to/source.cpp",
  "timestamp": "2018-05-16 16:13:18.717446360",
  "profile": {
    "time.clang-tidy.readability-function-size.wall": 1.0421266555786133e+00,
    "time.clang-tidy.readability-function-size.user": 9.2088400000005421e-01,
    "time.clang-tidy.readability-function-size.sys": 1.2418899999999974e-01
  }
  }

There is only one argument that controls profile storage:

* ``-store-check-profile=<prefix>``

  By default reports are printed in tabulated format to stderr. When this option
  is passed, these per-TU profiles are instead stored as JSON.
  If the prefix is not an absolute path, it is considered to be relative to the
  directory from where you have run :program:`clang-tidy`. All ``.`` and ``..``
  patterns in the path are collapsed, and symlinks are resolved.

  Example:
  Let's suppose you have a source file named ``example.cpp``, located in the
  ``/source`` directory. Only the input filename is used, not the full path
  to the source file. Additionally, it is prefixed with the current timestamp.

  * If you specify ``-store-check-profile=/tmp``, then the profile will be saved
    to ``/tmp/<ISO8601-like timestamp>-example.cpp.json``

  * If you run :program:`clang-tidy` from within ``/foo`` directory, and specify
    ``-store-check-profile=.``, then the profile will still be saved to
    ``/foo/<ISO8601-like timestamp>-example.cpp.json``
