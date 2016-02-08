==========
Clang-Tidy
==========

.. toctree::
   :maxdepth: 1

   checks/list


:program:`clang-tidy` is a clang-based C++ linter tool. Its purpose is to
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

  $ clang-tidy test.cpp -checks=-*,clang-analyzer-*,-clang-analyzer-alpha*

will disable all default checks (``-*``) and enable all ``clang-analyzer-*``
checks except for ``clang-analyzer-alpha*`` ones.

The ``-list-checks`` option lists all the enabled checks. When used without
``-checks=``, it shows checks enabled by default. Use ``-checks=*`` to see all
available checks or with any other value of ``-checks=`` to see which checks are
enabled by this value.

There are currently the following groups of checks:

* Checks related to the LLVM coding conventions have names starting with
  ``llvm-``.

* Checks related to the Google coding conventions have names starting with
  ``google-``.

* Checks named ``modernize-*`` advocate the usage of modern (currently "modern"
  means "C++11") language constructs.

* The ``readability-`` checks target readability-related issues that don't
  relate to any particular coding style.

* Checks with names starting with ``misc-`` the checks that we didn't have a
  better category for.

* Clang static analyzer checks are named starting with ``clang-analyzer-``.

Clang diagnostics are treated in a similar way as check diagnostics. Clang
diagnostics are displayed by clang-tidy and can be filtered out using
``-checks=`` option. However, the ``-checks=`` option does not affect
compilation arguments, so it can not turn on Clang warnings which are not
already turned on in build configuration. The ``-warnings-as-errors=`` option
upgrades any warnings emitted under the ``-checks=`` flag to errors (but it
does not enable any checks itself).

Clang diagnostics have check names starting with ``clang-diagnostic-``.
Diagnostics which have a corresponding warning option, are named
``clang-diagostic-<warning-option>``, e.g. Clang warning controlled by
``-Wliteral-conversion`` will be reported with check name
``clang-diagnostic-literal-conversion``.

The ``-fix`` flag instructs :program:`clang-tidy` to fix found errors if
supported by corresponding checks.

An overview of all the command-line options:

.. code-block:: console

  $ clang-tidy -help
  USAGE: clang-tidy [options] <source0> [... <sourceN>]

  OPTIONS:

  Generic Options:

    -help                        - Display available options (-help-hidden for more)
    -help-list                   - Display list of available options (-help-list-hidden for more)
    -version                     - Display the version of this program

  clang-tidy options:

    -analyze-temporary-dtors     - 
                                   Enable temporary destructor-aware analysis in
                                   clang-analyzer- checks.
                                   This option overrides the value read from a
                                   .clang-tidy file.
    -checks=<string>             - 
                                   Comma-separated list of globs with optional '-'
                                   prefix. Globs are processed in order of
                                   appearance in the list. Globs without '-'
                                   prefix add checks with matching names to the
                                   set, globs with the '-' prefix remove checks
                                   with matching names from the set of enabled
                                   checks.  This option's value is appended to the
                                   value of the 'Checks' option in .clang-tidy
                                   file, if any.
    -config=<string>             - 
                                   Specifies a configuration in YAML/JSON format:
                                     -config="{Checks: '*',
                                               CheckOptions: [{key: x,
                                                               value: y}]}"
                                   When the value is empty, clang-tidy will
                                   attempt to find a file named .clang-tidy for
                                   each source file in its parent directories.
    -dump-config                 - 
                                   Dumps configuration in the YAML format to
                                   stdout. This option can be used along with a
                                   file name (and '--' if the file is outside of a
                                   project with configured compilation database).
                                   The configuration used for this file will be
                                   printed.
                                   Use along with -checks=* to include
                                   configuration of all checks.
    -enable-check-profile        - 
                                   Enable per-check timing profiles, and print a
                                   report to stderr.
    -export-fixes=<filename>     - 
                                   YAML file to store suggested fixes in. The
                                   stored fixes can be applied to the input sorce
                                   code with clang-apply-replacements.
    -extra-arg=<string>          - Additional argument to append to the compiler command line
    -extra-arg-before=<string>   - Additional argument to prepend to the compiler command line
    -fix                         - 
                                   Apply suggested fixes. Without -fix-errors
                                   clang-tidy will bail out if any compilation
                                   errors were found.
    -fix-errors                  - 
                                   Apply suggested fixes even if compilation
                                   errors were found. If compiler errors have
                                   attached fix-its, clang-tidy will apply them as
                                   well.
    -header-filter=<string>      - 
                                   Regular expression matching the names of the
                                   headers to output diagnostics from. Diagnostics
                                   from the main file of each translation unit are
                                   always displayed.
                                   Can be used together with -line-filter.
                                   This option overrides the 'HeaderFilter' option
                                   in .clang-tidy file, if any.
    -line-filter=<string>        - 
                                   List of files with line ranges to filter the
                                   warnings. Can be used together with
                                   -header-filter. The format of the list is a
                                   JSON array of objects:
                                     [
                                       {"name":"file1.cpp","lines":[[1,3],[5,7]]},
                                       {"name":"file2.h"}
                                     ]
    -list-checks                 - 
                                   List all enabled checks and exit. Use with
                                   -checks=* to list all available checks.
    -p=<string>                  - Build path
    -system-headers              - Display the errors from system headers.
    -warnings-as-errors=<string> - 
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

      $ clang-tidy -dump-config - --
      ---
      Checks:          '-*,some-check'
      WarningsAsErrors: ''
      HeaderFilterRegex: ''
      AnalyzeTemporaryDtors: false
      User:            user
      CheckOptions:
        - key:             some-check.SomeOption
          value:           'some value'
      ...

.. _LibTooling: http://clang.llvm.org/docs/LibTooling.html
.. _How To Setup Tooling For LLVM: http://clang.llvm.org/docs/HowToSetupToolingForLLVM.html


Getting Involved
================

:program:`clang-tidy` has several own checks and can run Clang static analyzer
checks, but its power is in the ability to easily write custom checks.

Checks are organized in modules, which can be linked into :program:`clang-tidy`
with minimal or no code changes in clang-tidy.

Checks can plug the analysis on the preprocessor level using `PPCallbacks`_ or
on the AST level using `AST Matchers`_. When an error is found, checks can
report them in a way similar to how Clang diagnostics work. A fix-it hint can be
attached to a diagnostic message.

The interface provided by clang-tidy makes it easy to write useful and precise
checks in just a few lines of code. If you have an idea for a good check, the
rest of this document explains how to do this.

.. _AST Matchers: http://clang.llvm.org/docs/LibASTMatchers.html
.. _PPCallbacks: http://clang.llvm.org/doxygen/classclang_1_1PPCallbacks.html


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
  |-- tool/                         # Sources of the clang-tidy binary.
          ...
  test/clang-tidy/                  # Integration tests.
      ...
  unittests/clang-tidy/             # Unit tests.
  |-- ClangTidyTest.h
  |-- GoogleModuleTest.cpp
  |-- LLVMModuleTest.cpp
      ...


Writing a clang-tidy Check
--------------------------

So you have an idea of a useful check for :program:`clang-tidy`.

You need to decide which module the check belongs to. If the check verifies
conformance of the code to a certain coding style, it probably deserves a
separate module and a directory in ``clang-tidy/`` (there are LLVM and Google
modules already).

After choosing the module, you need to create a class for your check:

.. code-block:: c++

  #include "../ClangTidy.h"

  namespace clang {
  namespace tidy {

  class MyCheck : public ClangTidyCheck {
  };

  } // namespace tidy
  } // namespace clang

Next, you need to decide whether it should operate on the preprocessor level or
on the AST level. Let's imagine that we need to work with the AST in our check.
In this case we need to override two methods:

.. code-block:: c++

  ...
  class ExplicitConstructorCheck : public ClangTidyCheck {
  public:
    ExplicitConstructorCheck(StringRef Name, ClangTidyContext *Context)
        : ClangTidyCheck(Name, Context) {}
    void registerMatchers(ast_matchers::MatchFinder *Finder) override;
    void check(ast_matchers::MatchFinder::MatchResult &Result) override;
  };

Constructor of the check receives the ``Name`` and ``Context`` parameters, and
must forward them to the ``ClangTidyCheck`` constructor.

In the ``registerMatchers`` method we create an AST Matcher (see `AST Matchers`_
for more information) that will find the pattern in the AST that we want to
inspect. The results of the matching are passed to the ``check`` method, which
can further inspect them and report diagnostics.

.. code-block:: c++

  using namespace ast_matchers;

  void ExplicitConstructorCheck::registerMatchers(MatchFinder *Finder) {
    Finder->addMatcher(constructorDecl().bind("ctor"), this);
  }

  void ExplicitConstructorCheck::check(const MatchFinder::MatchResult &Result) {
    const CXXConstructorDecl *Ctor =
        Result.Nodes.getNodeAs<CXXConstructorDecl>("ctor");
    // Do not be confused: isExplicit means 'explicit' keyword is present,
    // isImplicit means that it's a compiler-generated constructor.
    if (Ctor->isOutOfLine() || Ctor->isExplicit() || Ctor->isImplicit())
      return;
    if (Ctor->getNumParams() == 0 || Ctor->getMinRequiredArguments() > 1)
      return;
    SourceLocation Loc = Ctor->getLocation();
    diag(Loc, "single-argument constructors must be explicit")
        << FixItHint::CreateInsertion(Loc, "explicit ");
  }

(The full code for this check resides in
``clang-tidy/google/ExplicitConstructorCheck.{h,cpp}``).


Registering your Check
----------------------

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
module is linked into the clang-tidy binary:

Add this near the ``ClangTidyModuleRegistry::Add<MyModule>`` variable:

.. code-block:: c++

  // This anchor is used to force the linker to link in the generated object file
  // and thus register the MyModule.
  volatile int MyModuleAnchorSource = 0;

And this to the main translation unit of the clang-tidy binary (or the binary
you link the ``clang-tidy`` library in) ``clang-tidy/tool/ClangTidyMain.cpp``:

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

.. code-block:: bash

  $ clang-tidy -config="{CheckOptions: [{key: a, value: b}, {key: x, value: y}]}" ...


Testing Checks
--------------

:program:`clang-tidy` checks can be tested using either unit tests or
`lit`_ tests. Unit tests may be more convenient to test complex replacements
with strict checks. `Lit`_ tests allow using partial text matching and regular
expressions which makes them more suitable for writing compact tests for
diagnostic messages.

The ``check_clang_tidy.py`` script provides an easy way to test both
diagnostic messages and fix-its. It filters out ``CHECK`` lines from the test
file, runs :program:`clang-tidy` and verifies messages and fixes with two
separate `FileCheck`_ invocations. To use the script, put a .cpp file with the
appropriate ``RUN`` line in the ``test/clang-tidy`` directory.  Use
``CHECK-MESSAGES:`` and ``CHECK-FIXES:`` lines to write checks against
diagnostic messages and fixed code.

It's advised to make the checks as specific as possible to avoid checks matching
to incorrect parts of the input. Use ``[[@LINE+X]]``/``[[@LINE-X]]``
substitutions and distinct function and variable names in the test code.

Here's an example of a test using the ``check_clang_tidy.py`` script:

.. code-block:: bash

  // RUN: %python %S/check_clang_tidy.py %s google-readability-casting %t

  void f(int a) {
    int b = (int)a;
    // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: redundant cast to the same type [google-readability-casting]
    // CHECK-FIXES: int b = a;
  }

.. _lit: http://llvm.org/docs/CommandGuide/lit.html
.. _FileCheck: http://llvm.org/docs/CommandGuide/FileCheck.html


Running clang-tidy on LLVM
--------------------------

To test a check it's best to try it out on a larger code base. LLVM and Clang
are the natural targets as you already have the source around. The most
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
  checkers. It may also be necessary to restrict the header files warnings are
  displayed from using the ``-header-filter`` flag. It has the same behavior
  as the corresponding :program:`clang-tidy` flag.

* To apply suggested fixes ``-fix`` can be passed as an argument. This gathers
  all changes in a temporary directory and applies them. Passing ``-format``
  will run clang-format over changed lines.

