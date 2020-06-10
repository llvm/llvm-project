=====================================
Cross Translation Unit (CTU) Analysis
=====================================

Normally, static analysis works in the boundary of one translation unit (TU).
However, with additional steps and configuration we can enable the analysis to inline the definition of a function from another TU.

.. contents::
   :local:

Manual CTU Analysis
-------------------

Let's consider these source files in our minimal example:

.. code-block:: cpp

  // main.cpp
  int foo();

  int main() {
    return 3 / foo();
  }

.. code-block:: cpp

  // foo.cpp
  int foo() {
    return 0;
  }

And a compilation database:

.. code-block:: bash

  [
    {
      "directory": "/path/to/your/project",
      "command": "clang++ -c foo.cpp -o foo.o",
      "file": "foo.cpp"
    },
    {
      "directory": "/path/to/your/project",
      "command": "clang++ -c main.cpp -o main.o",
      "file": "main.cpp"
    }
  ]

We'd like to analyze `main.cpp` and discover the division by zero bug.
In order to be able to inline the definition of `foo` from `foo.cpp` first we have to generate the `AST` (or `PCH`) file of `foo.cpp`:

.. code-block:: bash

  $ pwd $ /path/to/your/project
  $ clang++ -emit-ast -o foo.cpp.ast foo.cpp
  $ # Check that the .ast file is generated:
  $ ls
  compile_commands.json  foo.cpp.ast  foo.cpp  main.cpp
  $

The next step is to create a CTU index file which holds the `USR` name and location of external definitions in the source files:

.. code-block:: bash

  $ clang-extdef-mapping -p . foo.cpp
  c:@F@foo# /path/to/your/project/foo.cpp
  $ clang-extdef-mapping -p . foo.cpp > externalDefMap.txt

We have to modify `externalDefMap.txt` to contain the name of the `.ast` files instead of the source files:

.. code-block:: bash

  $ sed -i -e "s/.cpp/.cpp.ast/g" externalDefMap.txt

We still have to further modify the `externalDefMap.txt` file to contain relative paths:

.. code-block:: bash

  $ sed -i -e "s|$(pwd)/||g" externalDefMap.txt

Now everything is available for the CTU analysis.
We have to feed Clang with CTU specific extra arguments:

.. code-block:: bash

  $ pwd
  /path/to/your/project
  $ clang++ --analyze -Xclang -analyzer-config -Xclang experimental-enable-naive-ctu-analysis=true -Xclang -analyzer-config -Xclang ctu-dir=. -Xclang -analyzer-output=plist-multi-file main.cpp
  main.cpp:5:12: warning: Division by zero
    return 3 / foo();
           ~~^~~~~~~
  1 warning generated.
  $ # The plist file with the result is generated.
  $ ls
  compile_commands.json  externalDefMap.txt  foo.ast  foo.cpp  foo.cpp.ast  main.cpp  main.plist
  $

This manual procedure is error-prone and not scalable, therefore to analyze real projects it is recommended to use `CodeChecker` or `scan-build-py`.

Automated CTU Analysis with CodeChecker
---------------------------------------
The `CodeChecker <https://github.com/Ericsson/codechecker>`_ project fully supports automated CTU analysis with Clang.
Once we have set up the `PATH` environment variable and we activated the python `venv` then it is all it takes:

.. code-block:: bash

  $ CodeChecker analyze --ctu compile_commands.json -o reports
  [INFO 2019-07-16 17:21] - Pre-analysis started.
  [INFO 2019-07-16 17:21] - Collecting data for ctu analysis.
  [INFO 2019-07-16 17:21] - [1/2] foo.cpp
  [INFO 2019-07-16 17:21] - [2/2] main.cpp
  [INFO 2019-07-16 17:21] - Pre-analysis finished.
  [INFO 2019-07-16 17:21] - Starting static analysis ...
  [INFO 2019-07-16 17:21] - [1/2] clangsa analyzed foo.cpp successfully.
  [INFO 2019-07-16 17:21] - [2/2] clangsa analyzed main.cpp successfully.
  [INFO 2019-07-16 17:21] - ----==== Summary ====----
  [INFO 2019-07-16 17:21] - Successfully analyzed
  [INFO 2019-07-16 17:21] -   clangsa: 2
  [INFO 2019-07-16 17:21] - Total analyzed compilation commands: 2
  [INFO 2019-07-16 17:21] - ----=================----
  [INFO 2019-07-16 17:21] - Analysis finished.
  [INFO 2019-07-16 17:21] - To view results in the terminal use the "CodeChecker parse" command.
  [INFO 2019-07-16 17:21] - To store results use the "CodeChecker store" command.
  [INFO 2019-07-16 17:21] - See --help and the user guide for further options about parsing and storing the reports.
  [INFO 2019-07-16 17:21] - ----=================----
  [INFO 2019-07-16 17:21] - Analysis length: 0.659618854523 sec.
  $ ls
  compile_commands.json  foo.cpp  foo.cpp.ast  main.cpp  reports
  $ tree reports
  reports
  ├── compile_cmd.json
  ├── compiler_info.json
  ├── foo.cpp_53f6fbf7ab7ec9931301524b551959e2.plist
  ├── main.cpp_23db3d8df52ff0812e6e5a03071c8337.plist
  ├── metadata.json
  └── unique_compile_commands.json

  0 directories, 6 files
  $

The `plist` files contain the results of the analysis, which may be viewed with the regular analysis tools.
E.g. one may use `CodeChecker parse` to view the results in command line:

.. code-block:: bash

  $ CodeChecker parse reports
  [HIGH] /home/egbomrt/ctu_mini_raw_project/main.cpp:5:12: Division by zero [core.DivideZero]
    return 3 / foo();
             ^

  Found 1 defect(s) in main.cpp


  ----==== Summary ====----
  -----------------------
  Filename | Report count
  -----------------------
  main.cpp |            1
  -----------------------
  -----------------------
  Severity | Report count
  -----------------------
  HIGH     |            1
  -----------------------
  ----=================----
  Total number of reports: 1
  ----=================----

Or we can use `CodeChecker parse -e html` to export the results into HTML format:

.. code-block:: bash

  $ CodeChecker parse -e html -o html_out reports
  $ firefox html_out/index.html

Automated CTU Analysis with scan-build-py (don't do it)
-------------------------------------------------------
We actively develop CTU with CodeChecker as a "runner" script, `scan-build-py` is not actively developed for CTU.
`scan-build-py` has various errors and issues, expect it to work with the very basic projects only.

Example usage of scan-build-py:

.. code-block:: bash

  $ /your/path/to/llvm-project/clang/tools/scan-build-py/bin/analyze-build --ctu
  analyze-build: Run 'scan-view /tmp/scan-build-2019-07-17-17-53-33-810365-7fqgWk' to examine bug reports.
  $ /your/path/to/llvm-project/clang/tools/scan-view/bin/scan-view /tmp/scan-build-2019-07-17-17-53-33-810365-7fqgWk
  Starting scan-view at: http://127.0.0.1:8181
    Use Ctrl-C to exit.
  [6336:6431:0717/175357.633914:ERROR:browser_process_sub_thread.cc(209)] Waited 5 ms for network service
  Opening in existing browser session.
  ^C
  $
