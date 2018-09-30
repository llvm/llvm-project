==========
Clang-Tidy
==========

.. contents::

See also:

.. toctree::
   :maxdepth: 1

   The list of clang-tidy checks <checks/list>

:program:`clang-tidy` is a clang-based C++ "linter" tool. Its purpose is to
provide an extensible framework for diagnosing and fixing typical programming
errors, like style violations, interface misuse, or bugs that can be deduced via
static analysis. :program:`clang-tidy` is modular and provides a convenient
interface for writing new checks.


Using clang-tidy
================

:program:`clang-tidy` is a `LibTooling`_-based tool, and it's easier to work
with if you set up a compile command database for your project (for an example
of how to do this see `How To Setup Tooling For LLVM`_). You can also specify
compilation options on the command line after ``--``:

.. code-block:: console

  $ clang-tidy test.cpp -- -Imy_project/include -DMY_DEFINES ...

:program:`clang-tidy` has its own checks and can also run Clang static analyzer
checks. Each check has a name and the checks to run can be chosen using the
``-checks=`` option, which specifies a comma-separated list of positive and
negative (prefixed with ``-``) globs. Positive globs add subsets of checks,
negative globs remove them. For example,

.. code-block:: console

  $ clang-tidy test.cpp -checks=-*,clang-analyzer-*,-clang-analyzer-cplusplus*

will disable all default checks (``-*``) and enable all ``clang-analyzer-*``
checks except for ``clang-analyzer-cplusplus*`` ones.

The ``-list-checks`` option lists all the enabled checks. When used without
``-checks=``, it shows checks enabled by default. Use ``-checks=*`` to see all
available checks or with any other value of ``-checks=`` to see which checks are
enabled by this value.

.. _checks-groups-table:

There are currently the following groups of checks:

====================== =========================================================
Name prefix            Description
====================== =========================================================
``abseil-``            Checks related to Abseil library.
``android-``           Checks related to Android.
``boost-``             Checks related to Boost library.
``bugprone-``          Checks that target bugprone code constructs.
``cert-``              Checks related to CERT Secure Coding Guidelines.
``cppcoreguidelines-`` Checks related to C++ Core Guidelines.
``clang-analyzer-``    Clang Static Analyzer checks.
``fuchsia-``           Checks related to Fuchsia coding conventions.
``google-``            Checks related to Google coding conventions.
``hicpp-``             Checks related to High Integrity C++ Coding Standard.
``llvm-``              Checks related to the LLVM coding conventions.
``misc-``              Checks that we didn't have a better category for.
``modernize-``         Checks that advocate usage of modern (currently "modern"
                       means "C++11") language constructs.
``mpi-``               Checks related to MPI (Message Passing Interface).
``objc-``              Checks related to Objective-C coding conventions.
``performance-``       Checks that target performance-related issues.
``portability-``       Checks that target portability-related issues that don't
                       relate to any particular coding style.
``readability-``       Checks that target readability-related issues that don't
                       relate to any particular coding style.
``zircon-``            Checks related to Zircon kernel coding conventions.
====================== =========================================================

Clang diagnostics are treated in a similar way as check diagnostics. Clang
diagnostics are displayed by :program:`clang-tidy` and can be filtered out using
``-checks=`` option. However, the ``-checks=`` option does not affect
compilation arguments, so it can not turn on Clang warnings which are not
already turned on in build configuration. The ``-warnings-as-errors=`` option
upgrades any warnings emitted under the ``-checks=`` flag to errors (but it
does not enable any checks itself).

Clang diagnostics have check names starting with ``clang-diagnostic-``.
Diagnostics which have a corresponding warning option, are named
``clang-diagnostic-<warning-option>``, e.g. Clang warning controlled by
``-Wliteral-conversion`` will be reported with check name
``clang-diagnostic-literal-conversion``.

The ``-fix`` flag instructs :program:`clang-tidy` to fix found errors if
supported by corresponding checks.

An overview of all the command-line options:

.. code-block:: console

  $ clang-tidy --help
  USAGE: clang-tidy [options] <source0> [... <sourceN>]

  OPTIONS:

  Generic Options:

    -help                         - Display available options (-help-hidden for more)
    -help-list                    - Display list of available options (-help-list-hidden for more)
    -version                      - Display the version of this program

  clang-tidy options:

    -checks=<string>              -
                                    Comma-separated list of globs with optional '-'
                                    prefix. Globs are processed in order of
                                    appearance in the list. Globs without '-'
                                    prefix add checks with matching names to the
                                    set, globs with the '-' prefix remove checks
                                    with matching names from the set of enabled
                                    checks. This option's value is appended to the
                                    value of the 'Checks' option in .clang-tidy
                                    file, if any.
    -config=<string>              -
                                    Specifies a configuration in YAML/JSON format:
                                      -config="{Checks: '*',
                                                CheckOptions: [{key: x,
                                                                value: y}]}"
                                    When the value is empty, clang-tidy will
                                    attempt to find a file named .clang-tidy for
                                    each source file in its parent directories.
    -dump-config                  -
                                    Dumps configuration in the YAML format to
                                    stdout. This option can be used along with a
                                    file name (and '--' if the file is outside of a
                                    project with configured compilation database).
                                    The configuration used for this file will be
                                    printed.
                                    Use along with -checks=* to include
                                    configuration of all checks.
    -enable-check-profile         -
                                    Enable per-check timing profiles, and print a
                                    report to stderr.
    -explain-config               -
                                    For each enabled check explains, where it is
                                    enabled, i.e. in clang-tidy binary, command
                                    line or a specific configuration file.
    -export-fixes=<filename>      -
                                    YAML file to store suggested fixes in. The
                                    stored fixes can be applied to the input source
                                    code with clang-apply-replacements.
    -extra-arg=<string>           - Additional argument to append to the compiler command line
    -extra-arg-before=<string>    - Additional argument to prepend to the compiler command line
    -fix                          -
                                    Apply suggested fixes. Without -fix-errors
                                    clang-tidy will bail out if any compilation
                                    errors were found.
    -fix-errors                   -
                                    Apply suggested fixes even if compilation
                                    errors were found. If compiler errors have
                                    attached fix-its, clang-tidy will apply them as
                                    well.
    -format-style=<string>        -
                                    Style for formatting code around applied fixes:
                                      - 'none' (default) turns off formatting
                                      - 'file' (literally 'file', not a placeholder)
                                        uses .clang-format file in the closest parent
                                        directory
                                      - '{ <json> }' specifies options inline, e.g.
                                        -format-style='{BasedOnStyle: llvm, IndentWidth: 8}'
                                      - 'llvm', 'google', 'webkit', 'mozilla'
                                    See clang-format documentation for the up-to-date
                                    information about formatting styles and options.
                                    This option overrides the 'FormatStyle` option in
                                    .clang-tidy file, if any.
    -header-filter=<string>       -
                                    Regular expression matching the names of the
                                    headers to output diagnostics from. Diagnostics
                                    from the main file of each translation unit are
                                    always displayed.
                                    Can be used together with -line-filter.
                                    This option overrides the 'HeaderFilter' option
                                    in .clang-tidy file, if any.
    -line-filter=<string>         -
                                    List of files with line ranges to filter the
                                    warnings. Can be used together with
                                    -header-filter. The format of the list is a
                                    JSON array of objects:
                                      [
                                        {"name":"file1.cpp","lines":[[1,3],[5,7]]},
                                        {"name":"file2.h"}
                                      ]
    -list-checks                  -
                                    List all enabled checks and exit. Use with
                                    -checks=* to list all available checks.
    -p=<string>                   - Build path
    -quiet                        -
                                    Run clang-tidy in quiet mode. This suppresses
                                    printing statistics about ignored warnings and
                                    warnings treated as errors if the respective
                                    options are specified.
    -store-check-profile=<prefix> -
                                    By default reports are printed in tabulated
                                    format to stderr. When this option is passed,
                                    these per-TU profiles are instead stored as JSON.
    -system-headers               - Display the errors from system headers.
    -vfsoverlay=<filename>        -
                                    Overlay the virtual filesystem described by file
                                    over the real file system.
    -warnings-as-errors=<string>  -
                                    Upgrades warnings to errors. Same format as
                                    '-checks'.
                                    This option's value is appended to the value of
                                    the 'WarningsAsErrors' option in .clang-tidy
                                    file, if any.

  -p <build-path> is used to read a compile command database.

          For example, it can be a CMake build directory in which a file named
          compile_commands.json exists (use -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
          CMake option to get this output). When no build path is specified,
          a search for compile_commands.json will be attempted through all
          parent paths of the first input file . See:
          http://clang.llvm.org/docs/HowToSetupToolingForLLVM.html for an
          example of setting up Clang Tooling on a source tree.

  <source0> ... specify the paths of source files. These paths are
          looked up in the compile command database. If the path of a file is
          absolute, it needs to point into CMake's source tree. If the path is
          relative, the current working directory needs to be in the CMake
          source tree and the file must be in a subdirectory of the current
          working directory. "./" prefixes in the relative files will be
          automatically removed, but the rest of a relative path must be a
          suffix of a path in the compile command database.


  Configuration files:
    clang-tidy attempts to read configuration for each source file from a
    .clang-tidy file located in the closest parent directory of the source
    file. If any configuration options have a corresponding command-line
    option, command-line option takes precedence. The effective
    configuration can be inspected using -dump-config:

      $ clang-tidy -dump-config
      ---
      Checks:          '-*,some-check'
      WarningsAsErrors: ''
      HeaderFilterRegex: ''
      FormatStyle:     none
      User:            user
      CheckOptions:
        - key:             some-check.SomeOption
          value:           'some value'
      ...

:program:`clang-tidy` diagnostics are intended to call out code that does
not adhere to a coding standard, or is otherwise problematic in some way.
However, if it is known that the code is correct, the check-specific ways
to silence the diagnostics could be used, if they are available (e.g. 
bugprone-use-after-move can be silenced by re-initializing the variable after 
it has been moved out, misc-string-integer-assignment can be suppressed by 
explicitly casting the integer to char, readability-implicit-bool-conversion
can also be suppressed by using explicit casts, etc.). If they are not 
available or if changing the semantics of the code is not desired, 
the ``NOLINT`` or ``NOLINTNEXTLINE`` comments can be used instead. For example:

.. code-block:: c++

  class Foo
  {
    // Silent all the diagnostics for the line
    Foo(int param); // NOLINT

    // Silent only the specified checks for the line
    Foo(double param); // NOLINT(google-explicit-constructor, google-runtime-int)

    // Silent only the specified diagnostics for the next line
    // NOLINTNEXTLINE(google-explicit-constructor, google-runtime-int)
    Foo(bool param); 
  };

The formal syntax of ``NOLINT``/``NOLINTNEXTLINE`` is the following:

.. parsed-literal::

  lint-comment:
    lint-command
    lint-command lint-args

  lint-args:
    **(** check-name-list **)**

  check-name-list:
    *check-name*
    check-name-list **,** *check-name*

  lint-command:
    **NOLINT**
    **NOLINTNEXTLINE**

Note that whitespaces between ``NOLINT``/``NOLINTNEXTLINE`` and the opening
parenthesis are not allowed (in this case the comment will be treated just as
``NOLINT``/``NOLINTNEXTLINE``), whereas in check names list (inside
the parenthesis) whitespaces can be used and will be ignored.

.. _LibTooling: http://clang.llvm.org/docs/LibTooling.html
.. _How To Setup Tooling For LLVM: http://clang.llvm.org/docs/HowToSetupToolingForLLVM.html


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
  * :program:`clang-query` is invaluable for interactive prototyping of AST
    matchers and exploration of the Clang AST;
  * `clang-check`_ with the ``-ast-dump`` (and optionally ``-ast-dump-filter``)
    provides a convenient way to dump AST of a C++ program.


.. _AST Matchers: http://clang.llvm.org/docs/LibASTMatchers.html
.. _PPCallbacks: http://clang.llvm.org/doxygen/classclang_1_1PPCallbacks.html
.. _clang-check: http://clang.llvm.org/docs/ClangCheck.html


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
the LLVM System`_, `Using Clang Tools`_ and `How To Setup Tooling For LLVM`_
documents to check out and build LLVM, Clang and Clang Extra Tools with CMake.

Once you are done, change to the ``llvm/tools/clang/tools/extra`` directory, and
let's start!

.. _Getting Started with the LLVM System: http://llvm.org/docs/GettingStarted.html
.. _Using Clang Tools: http://clang.llvm.org/docs/ClangTools.html


The Directory Structure
-----------------------

:program:`clang-tidy` source code resides in the
``llvm/tools/clang/tools/extra`` directory and is structured as follows:

::

  clang-tidy/                       # Clang-tidy core.
  |-- ClangTidy.h                   # Interfaces for users and checks.
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

.. _Getting Started with LLVM: http://llvm.org/docs/GettingStarted.html
.. _LLVM Coding Standards: http://llvm.org/docs/CodingStandards.html
.. _LLVM Phabricator: http://llvm.org/docs/Phabricator.html

Next, you need to decide which module the check belongs to. Modules
are located in subdirectories of `clang-tidy/
<http://reviews.llvm.org/diffusion/L/browse/clang-tools-extra/trunk/clang-tidy/>`_
and contain checks targeting a certain aspect of code quality (performance,
readability, etc.), certain coding style or standard (Google, LLVM, CERT, etc.)
or a widely used API (e.g. MPI). Their names are same as user-facing check
groups names described :ref:`above <checks-groups-table>`.

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

  #include "../ClangTidy.h"

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
    if (MatchedDecl->getName().startswith("awesome_"))
      return;
    diag(MatchedDecl->getLocation(), "function %0 is insufficiently awesome")
        << MatchedDecl
        << FixItHint::CreateInsertion(MatchedDecl->getLocation(), "awesome_");
  }

(If you want to see an example of a useful check, look at
`clang-tidy/google/ExplicitConstructorCheck.h
<http://reviews.llvm.org/diffusion/L/browse/clang-tools-extra/trunk/clang-tidy/google/ExplicitConstructorCheck.h>`_
and `clang-tidy/google/ExplicitConstructorCheck.cpp
<http://reviews.llvm.org/diffusion/L/browse/clang-tools-extra/trunk/clang-tidy/google/ExplicitConstructorCheck.cpp>`_).


Registering your Check
----------------------

(The ``add_new_check.py`` takes care of registering the check in an existing
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
    - key: my-check.SomeOption1
      value: 123
    - key: my-check.SomeOption2
      value: 'some other value'

If you need to specify check options on a command line, you can use the inline
YAML format:

.. code-block:: console

  $ clang-tidy -config="{CheckOptions: [{key: a, value: b}, {key: x, value: y}]}" ...


Testing Checks
--------------

To run tests for :program:`clang-tidy` use the command:

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
source code is at `test/clang-tidy/google-readability-casting.cpp`_):

.. code-block:: c++

  // RUN: %check_clang_tidy %s google-readability-casting %t

  void f(int a) {
    int b = (int)a;
    // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: redundant cast to the same type [google-readability-casting]
    // CHECK-FIXES: int b = a;
  }

To check more than one scenario in the same test file use 
``-check-suffix=SUFFIX-NAME`` on ``check_clang_tidy.py`` command line.
With ``-check-suffix=SUFFIX-NAME`` you need to replace your ``CHECK-*`` 
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

.. _lit: http://llvm.org/docs/CommandGuide/lit.html
.. _FileCheck: http://llvm.org/docs/CommandGuide/FileCheck.html
.. _test/clang-tidy/google-readability-casting.cpp: http://reviews.llvm.org/diffusion/L/browse/clang-tools-extra/trunk/test/clang-tidy/google-readability-casting.cpp


Running clang-tidy on LLVM
--------------------------

To test a check it's best to try it out on a larger code base. LLVM and Clang
are the natural targets as you already have the source code around. The most
convenient way to run :program:`clang-tidy` is with a compile command database;
CMake can automatically generate one, for a description of how to enable it see
`How To Setup Tooling For LLVM`_. Once ``compile_commands.json`` is in place and
a working version of :program:`clang-tidy` is in ``PATH`` the entire code base
can be analyzed with ``clang-tidy/tool/run-clang-tidy.py``. The script executes
:program:`clang-tidy` with the default set of checks on every translation unit
in the compile command database and displays the resulting warnings and errors.
The script provides multiple configuration flags.

* The default set of checks can be overridden using the ``-checks`` argument,
  taking the identical format as :program:`clang-tidy` does. For example
  ``-checks=-*,modernize-use-override`` will run the ``modernize-use-override``
  check only.

* To restrict the files examined you can provide one or more regex arguments
  that the file names are matched against.
  ``run-clang-tidy.py clang-tidy/.*Check\.cpp`` will only analyze clang-tidy
  checks. It may also be necessary to restrict the header files warnings are
  displayed from using the ``-header-filter`` flag. It has the same behavior
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

  $ clang-tidy -enable-check-profile -store-check-profile=.  -checks=-*,readability-function-size source.cpp
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
