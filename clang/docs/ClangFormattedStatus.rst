.. raw:: html

      <style type="text/css">
        .none { background-color: #FFCC99 }
        .part { background-color: #FFFF99 }
        .good { background-color: #2CCCFF }
        .total { font-weight: bold; }
      </style>

.. role:: none
.. role:: part
.. role:: good
.. role:: total

======================
Clang Formatted Status
======================

:doc:`ClangFormattedStatus` describes the state of LLVM source
tree in terms of conformance to :doc:`ClangFormat` as of: December 04, 2020 17:56:14 (`840e651dc6d <https://github.com/llvm/llvm-project/commit/840e651dc6d>`_).


.. list-table:: LLVM Clang-Format Status
   :widths: 50 25 25 25 25
   :header-rows: 1

   * - Directory
     - Total Files
     - Formatted Files
     - Unformatted Files
     - % Complete
   * - clang/bindings/python/tests/cindex/INPUTS
     - `5`
     - `3`
     - `2`
     - :part:`60%`
   * - clang/docs/analyzer/checkers
     - `2`
     - `0`
     - `2`
     - :none:`0%`
   * - clang/examples/AnnotateFunctions
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - clang/examples/Attribute
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - clang/examples/CallSuperAttribute
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - clang/examples/clang-interpreter
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - clang/examples/PrintFunctionNames
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - clang/include/clang/Analysis
     - `15`
     - `4`
     - `11`
     - :part:`26%`
   * - clang/include/clang/Analysis/Analyses
     - `14`
     - `2`
     - `12`
     - :part:`14%`
   * - clang/include/clang/Analysis/DomainSpecific
     - `2`
     - `0`
     - `2`
     - :none:`0%`
   * - clang/include/clang/Analysis/FlowSensitive
     - `2`
     - `1`
     - `1`
     - :part:`50%`
   * - clang/include/clang/Analysis/Support
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - clang/include/clang/APINotes
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - clang/include/clang/ARCMigrate
     - `3`
     - `0`
     - `3`
     - :none:`0%`
   * - clang/include/clang/AST
     - `114`
     - `21`
     - `93`
     - :part:`18%`
   * - clang/include/clang/ASTMatchers
     - `5`
     - `0`
     - `5`
     - :none:`0%`
   * - clang/include/clang/ASTMatchers/Dynamic
     - `4`
     - `1`
     - `3`
     - :part:`25%`
   * - clang/include/clang/Basic
     - `78`
     - `27`
     - `51`
     - :part:`34%`
   * - clang/include/clang/CodeGen
     - `9`
     - `0`
     - `9`
     - :none:`0%`
   * - clang/include/clang/CrossTU
     - `2`
     - `1`
     - `1`
     - :part:`50%`
   * - clang/include/clang/DirectoryWatcher
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - clang/include/clang/Driver
     - `17`
     - `5`
     - `12`
     - :part:`29%`
   * - clang/include/clang/Edit
     - `5`
     - `1`
     - `4`
     - :part:`20%`
   * - clang/include/clang/Format
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - clang/include/clang/Frontend
     - `28`
     - `7`
     - `21`
     - :part:`25%`
   * - clang/include/clang/FrontendTool
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - clang/include/clang/Index
     - `7`
     - `2`
     - `5`
     - :part:`28%`
   * - clang/include/clang/IndexSerialization
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - clang/include/clang/Lex
     - `29`
     - `4`
     - `25`
     - :part:`13%`
   * - clang/include/clang/Parse
     - `5`
     - `2`
     - `3`
     - :part:`40%`
   * - clang/include/clang/Rewrite/Core
     - `6`
     - `0`
     - `6`
     - :none:`0%`
   * - clang/include/clang/Rewrite/Frontend
     - `4`
     - `0`
     - `4`
     - :none:`0%`
   * - clang/include/clang/Sema
     - `32`
     - `3`
     - `29`
     - :part:`9%`
   * - clang/include/clang/Serialization
     - `14`
     - `2`
     - `12`
     - :part:`14%`
   * - clang/include/clang/StaticAnalyzer/Checkers
     - `4`
     - `1`
     - `3`
     - :part:`25%`
   * - clang/include/clang/StaticAnalyzer/Core
     - `5`
     - `1`
     - `4`
     - :part:`20%`
   * - clang/include/clang/StaticAnalyzer/Core/BugReporter
     - `4`
     - `1`
     - `3`
     - :part:`25%`
   * - clang/include/clang/StaticAnalyzer/Core/PathSensitive
     - `36`
     - `10`
     - `26`
     - :part:`27%`
   * - clang/include/clang/StaticAnalyzer/Frontend
     - `5`
     - `2`
     - `3`
     - :part:`40%`
   * - clang/include/clang/Testing
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - clang/include/clang/Tooling
     - `16`
     - `9`
     - `7`
     - :part:`56%`
   * - clang/include/clang/Tooling/ASTDiff
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - clang/include/clang/Tooling/Core
     - `2`
     - `0`
     - `2`
     - :none:`0%`
   * - clang/include/clang/Tooling/DependencyScanning
     - `5`
     - `4`
     - `1`
     - :part:`80%`
   * - clang/include/clang/Tooling/Inclusions
     - `2`
     - `0`
     - `2`
     - :none:`0%`
   * - clang/include/clang/Tooling/Refactoring
     - `15`
     - `13`
     - `2`
     - :part:`86%`
   * - clang/include/clang/Tooling/Refactoring/Extract
     - `2`
     - `1`
     - `1`
     - :part:`50%`
   * - clang/include/clang/Tooling/Refactoring/Rename
     - `6`
     - `5`
     - `1`
     - :part:`83%`
   * - clang/include/clang/Tooling/Syntax
     - `5`
     - `5`
     - `0`
     - :good:`100%`
   * - clang/include/clang/Tooling/Transformer
     - `8`
     - `7`
     - `1`
     - :part:`87%`
   * - clang/include/clang-c
     - `10`
     - `3`
     - `7`
     - :part:`30%`
   * - clang/INPUTS
     - `2`
     - `0`
     - `2`
     - :none:`0%`
   * - clang/lib/Analysis
     - `26`
     - `3`
     - `23`
     - :part:`11%`
   * - clang/lib/Analysis/plugins/CheckerDependencyHandling
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - clang/lib/Analysis/plugins/CheckerOptionHandling
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - clang/lib/Analysis/plugins/SampleAnalyzer
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - clang/lib/APINotes
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - clang/lib/ARCMigrate
     - `22`
     - `0`
     - `22`
     - :none:`0%`
   * - clang/lib/AST
     - `80`
     - `2`
     - `78`
     - :part:`2%`
   * - clang/lib/AST/Interp
     - `44`
     - `19`
     - `25`
     - :part:`43%`
   * - clang/lib/ASTMatchers
     - `3`
     - `0`
     - `3`
     - :none:`0%`
   * - clang/lib/ASTMatchers/Dynamic
     - `6`
     - `1`
     - `5`
     - :part:`16%`
   * - clang/lib/Basic
     - `35`
     - `10`
     - `25`
     - :part:`28%`
   * - clang/lib/Basic/Targets
     - `48`
     - `24`
     - `24`
     - :part:`50%`
   * - clang/lib/CodeGen
     - `91`
     - `11`
     - `80`
     - :part:`12%`
   * - clang/lib/CrossTU
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - clang/lib/DirectoryWatcher
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - clang/lib/DirectoryWatcher/default
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - clang/lib/DirectoryWatcher/linux
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - clang/lib/DirectoryWatcher/mac
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - clang/lib/DirectoryWatcher/windows
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - clang/lib/Driver
     - `16`
     - `3`
     - `13`
     - :part:`18%`
   * - clang/lib/Driver/ToolChains
     - `85`
     - `29`
     - `56`
     - :part:`34%`
   * - clang/lib/Driver/ToolChains/Arch
     - `18`
     - `4`
     - `14`
     - :part:`22%`
   * - clang/lib/Edit
     - `3`
     - `0`
     - `3`
     - :none:`0%`
   * - clang/lib/Format
     - `31`
     - `31`
     - `0`
     - :good:`100%`
   * - clang/lib/Format/old
     - `2`
     - `1`
     - `1`
     - :part:`50%`
   * - clang/lib/Frontend
     - `33`
     - `4`
     - `29`
     - :part:`12%`
   * - clang/lib/Frontend/Rewrite
     - `8`
     - `0`
     - `8`
     - :none:`0%`
   * - clang/lib/FrontendTool
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - clang/lib/Headers
     - `136`
     - `12`
     - `124`
     - :part:`8%`
   * - clang/lib/Headers/openmp_wrappers
     - `5`
     - `5`
     - `0`
     - :good:`100%`
   * - clang/lib/Headers/ppc_wrappers
     - `7`
     - `2`
     - `5`
     - :part:`28%`
   * - clang/lib/Index
     - `12`
     - `2`
     - `10`
     - :part:`16%`
   * - clang/lib/IndexSerialization
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - clang/lib/Lex
     - `23`
     - `1`
     - `22`
     - :part:`4%`
   * - clang/lib/Parse
     - `15`
     - `1`
     - `14`
     - :part:`6%`
   * - clang/lib/Rewrite
     - `5`
     - `0`
     - `5`
     - :none:`0%`
   * - clang/lib/Sema
     - `55`
     - `4`
     - `51`
     - :part:`7%`
   * - clang/lib/Serialization
     - `17`
     - `1`
     - `16`
     - :part:`5%`
   * - clang/lib/StaticAnalyzer/Checkers
     - `117`
     - `16`
     - `101`
     - :part:`13%`
   * - clang/lib/StaticAnalyzer/Checkers/cert
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - clang/lib/StaticAnalyzer/Checkers/MPI-Checker
     - `6`
     - `0`
     - `6`
     - :none:`0%`
   * - clang/lib/StaticAnalyzer/Checkers/RetainCountChecker
     - `4`
     - `0`
     - `4`
     - :none:`0%`
   * - clang/lib/StaticAnalyzer/Checkers/UninitializedObject
     - `3`
     - `1`
     - `2`
     - :part:`33%`
   * - clang/lib/StaticAnalyzer/Checkers/WebKit
     - `10`
     - `8`
     - `2`
     - :part:`80%`
   * - clang/lib/StaticAnalyzer/Core
     - `46`
     - `9`
     - `37`
     - :part:`19%`
   * - clang/lib/StaticAnalyzer/Frontend
     - `8`
     - `3`
     - `5`
     - :part:`37%`
   * - clang/lib/Testing
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - clang/lib/Tooling
     - `15`
     - `6`
     - `9`
     - :part:`40%`
   * - clang/lib/Tooling/ASTDiff
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - clang/lib/Tooling/Core
     - `2`
     - `0`
     - `2`
     - :none:`0%`
   * - clang/lib/Tooling/DependencyScanning
     - `5`
     - `2`
     - `3`
     - :part:`40%`
   * - clang/lib/Tooling/Inclusions
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - clang/lib/Tooling/Refactoring
     - `5`
     - `3`
     - `2`
     - :part:`60%`
   * - clang/lib/Tooling/Refactoring/Extract
     - `2`
     - `1`
     - `1`
     - :part:`50%`
   * - clang/lib/Tooling/Refactoring/Rename
     - `5`
     - `2`
     - `3`
     - :part:`40%`
   * - clang/lib/Tooling/Syntax
     - `7`
     - `7`
     - `0`
     - :good:`100%`
   * - clang/lib/Tooling/Transformer
     - `7`
     - `4`
     - `3`
     - :part:`57%`
   * - clang/tools/apinotes-test
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - clang/tools/arcmt-test
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - clang/tools/c-index-test
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - clang/tools/clang-check
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - clang/tools/clang-diff
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - clang/tools/clang-extdef-mapping
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - clang/tools/clang-format
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - clang/tools/clang-format/fuzzer
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - clang/tools/clang-fuzzer
     - `6`
     - `4`
     - `2`
     - :part:`66%`
   * - clang/tools/clang-fuzzer/fuzzer-initialize
     - `2`
     - `0`
     - `2`
     - :none:`0%`
   * - clang/tools/clang-fuzzer/handle-cxx
     - `2`
     - `0`
     - `2`
     - :none:`0%`
   * - clang/tools/clang-fuzzer/handle-llvm
     - `3`
     - `1`
     - `2`
     - :part:`33%`
   * - clang/tools/clang-fuzzer/proto-to-cxx
     - `5`
     - `0`
     - `5`
     - :none:`0%`
   * - clang/tools/clang-fuzzer/proto-to-llvm
     - `3`
     - `0`
     - `3`
     - :none:`0%`
   * - clang/tools/clang-import-test
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - clang/tools/clang-offload-bundler
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - clang/tools/clang-offload-wrapper
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - clang/tools/clang-refactor
     - `4`
     - `4`
     - `0`
     - :good:`100%`
   * - clang/tools/clang-rename
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - clang/tools/clang-scan-deps
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - clang/tools/clang-shlib
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - clang/tools/diagtool
     - `9`
     - `0`
     - `9`
     - :none:`0%`
   * - clang/tools/driver
     - `4`
     - `1`
     - `3`
     - :part:`25%`
   * - clang/tools/libclang
     - `35`
     - `6`
     - `29`
     - :part:`17%`
   * - clang/tools/scan-build-py/tests/functional/src/include
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - clang/unittests/Analysis
     - `5`
     - `2`
     - `3`
     - :part:`40%`
   * - clang/unittests/AST
     - `27`
     - `6`
     - `21`
     - :part:`22%`
   * - clang/unittests/ASTMatchers
     - `6`
     - `3`
     - `3`
     - :part:`50%`
   * - clang/unittests/ASTMatchers/Dynamic
     - `3`
     - `0`
     - `3`
     - :none:`0%`
   * - clang/unittests/Basic
     - `6`
     - `2`
     - `4`
     - :part:`33%`
   * - clang/unittests/CodeGen
     - `7`
     - `1`
     - `6`
     - :part:`14%`
   * - clang/unittests/CrossTU
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - clang/unittests/DirectoryWatcher
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - clang/unittests/Driver
     - `5`
     - `1`
     - `4`
     - :part:`20%`
   * - clang/unittests/Format
     - `21`
     - `20`
     - `1`
     - :part:`95%`
   * - clang/unittests/Frontend
     - `10`
     - `6`
     - `4`
     - :part:`60%`
   * - clang/unittests/Index
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - clang/unittests/Lex
     - `6`
     - `1`
     - `5`
     - :part:`16%`
   * - clang/unittests/libclang
     - `2`
     - `0`
     - `2`
     - :none:`0%`
   * - clang/unittests/libclang/CrashTests
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - clang/unittests/Rename
     - `6`
     - `0`
     - `6`
     - :none:`0%`
   * - clang/unittests/Rewrite
     - `2`
     - `1`
     - `1`
     - :part:`50%`
   * - clang/unittests/Sema
     - `3`
     - `2`
     - `1`
     - :part:`66%`
   * - clang/unittests/Serialization
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - clang/unittests/StaticAnalyzer
     - `12`
     - `5`
     - `7`
     - :part:`41%`
   * - clang/unittests/Tooling
     - `29`
     - `8`
     - `21`
     - :part:`27%`
   * - clang/unittests/Tooling/RecursiveASTVisitorTests
     - `30`
     - `13`
     - `17`
     - :part:`43%`
   * - clang/unittests/Tooling/Syntax
     - `7`
     - `7`
     - `0`
     - :good:`100%`
   * - clang/utils/perf-training/cxx
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - clang/utils/TableGen
     - `21`
     - `3`
     - `18`
     - :part:`14%`
   * - clang-tools-extra/clang-apply-replacements/include/clang-apply-replacements/Tooling
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - clang-tools-extra/clang-apply-replacements/lib/Tooling
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - clang-tools-extra/clang-apply-replacements/tool
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - clang-tools-extra/clang-change-namespace
     - `2`
     - `0`
     - `2`
     - :none:`0%`
   * - clang-tools-extra/clang-change-namespace/tool
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - clang-tools-extra/clang-doc
     - `17`
     - `16`
     - `1`
     - :part:`94%`
   * - clang-tools-extra/clang-doc/tool
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - clang-tools-extra/clang-include-fixer
     - `13`
     - `7`
     - `6`
     - :part:`53%`
   * - clang-tools-extra/clang-include-fixer/find-all-symbols
     - `17`
     - `13`
     - `4`
     - :part:`76%`
   * - clang-tools-extra/clang-include-fixer/find-all-symbols/tool
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - clang-tools-extra/clang-include-fixer/plugin
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - clang-tools-extra/clang-include-fixer/tool
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - clang-tools-extra/clang-move
     - `4`
     - `1`
     - `3`
     - :part:`25%`
   * - clang-tools-extra/clang-move/tool
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - clang-tools-extra/clang-query
     - `5`
     - `4`
     - `1`
     - :part:`80%`
   * - clang-tools-extra/clang-query/tool
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - clang-tools-extra/clang-reorder-fields
     - `2`
     - `1`
     - `1`
     - :part:`50%`
   * - clang-tools-extra/clang-reorder-fields/tool
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - clang-tools-extra/clang-tidy
     - `18`
     - `11`
     - `7`
     - :part:`61%`
   * - clang-tools-extra/clang-tidy/abseil
     - `40`
     - `28`
     - `12`
     - :part:`70%`
   * - clang-tools-extra/clang-tidy/altera
     - `5`
     - `3`
     - `2`
     - :part:`60%`
   * - clang-tools-extra/clang-tidy/android
     - `33`
     - `23`
     - `10`
     - :part:`69%`
   * - clang-tools-extra/clang-tidy/boost
     - `3`
     - `3`
     - `0`
     - :good:`100%`
   * - clang-tools-extra/clang-tidy/bugprone
     - `111`
     - `89`
     - `22`
     - :part:`80%`
   * - clang-tools-extra/clang-tidy/cert
     - `29`
     - `27`
     - `2`
     - :part:`93%`
   * - clang-tools-extra/clang-tidy/concurrency
     - `3`
     - `2`
     - `1`
     - :part:`66%`
   * - clang-tools-extra/clang-tidy/cppcoreguidelines
     - `43`
     - `39`
     - `4`
     - :part:`90%`
   * - clang-tools-extra/clang-tidy/darwin
     - `5`
     - `2`
     - `3`
     - :part:`40%`
   * - clang-tools-extra/clang-tidy/fuchsia
     - `15`
     - `9`
     - `6`
     - :part:`60%`
   * - clang-tools-extra/clang-tidy/google
     - `33`
     - `22`
     - `11`
     - :part:`66%`
   * - clang-tools-extra/clang-tidy/hicpp
     - `9`
     - `7`
     - `2`
     - :part:`77%`
   * - clang-tools-extra/clang-tidy/linuxkernel
     - `3`
     - `2`
     - `1`
     - :part:`66%`
   * - clang-tools-extra/clang-tidy/llvm
     - `11`
     - `10`
     - `1`
     - :part:`90%`
   * - clang-tools-extra/clang-tidy/llvmlibc
     - `7`
     - `7`
     - `0`
     - :good:`100%`
   * - clang-tools-extra/clang-tidy/misc
     - `29`
     - `25`
     - `4`
     - :part:`86%`
   * - clang-tools-extra/clang-tidy/modernize
     - `67`
     - `48`
     - `19`
     - :part:`71%`
   * - clang-tools-extra/clang-tidy/mpi
     - `5`
     - `4`
     - `1`
     - :part:`80%`
   * - clang-tools-extra/clang-tidy/objc
     - `15`
     - `10`
     - `5`
     - :part:`66%`
   * - clang-tools-extra/clang-tidy/openmp
     - `5`
     - `5`
     - `0`
     - :good:`100%`
   * - clang-tools-extra/clang-tidy/performance
     - `29`
     - `22`
     - `7`
     - :part:`75%`
   * - clang-tools-extra/clang-tidy/plugin
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - clang-tools-extra/clang-tidy/portability
     - `5`
     - `3`
     - `2`
     - :part:`60%`
   * - clang-tools-extra/clang-tidy/readability
     - `79`
     - `64`
     - `15`
     - :part:`81%`
   * - clang-tools-extra/clang-tidy/tool
     - `3`
     - `2`
     - `1`
     - :part:`66%`
   * - clang-tools-extra/clang-tidy/utils
     - `35`
     - `28`
     - `7`
     - :part:`80%`
   * - clang-tools-extra/clang-tidy/zircon
     - `3`
     - `3`
     - `0`
     - :good:`100%`
   * - clang-tools-extra/clangd
     - `84`
     - `70`
     - `14`
     - :part:`83%`
   * - clang-tools-extra/clangd/benchmarks
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - clang-tools-extra/clangd/benchmarks/CompletionModel
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - clang-tools-extra/clangd/fuzzer
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - clang-tools-extra/clangd/index
     - `39`
     - `36`
     - `3`
     - :part:`92%`
   * - clang-tools-extra/clangd/index/dex
     - `9`
     - `8`
     - `1`
     - :part:`88%`
   * - clang-tools-extra/clangd/index/dex/dexp
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - clang-tools-extra/clangd/index/remote
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - clang-tools-extra/clangd/index/remote/marshalling
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - clang-tools-extra/clangd/index/remote/server
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - clang-tools-extra/clangd/index/remote/unimplemented
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - clang-tools-extra/clangd/indexer
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - clang-tools-extra/clangd/refactor
     - `4`
     - `4`
     - `0`
     - :good:`100%`
   * - clang-tools-extra/clangd/refactor/tweaks
     - `14`
     - `11`
     - `3`
     - :part:`78%`
   * - clang-tools-extra/clangd/support
     - `22`
     - `22`
     - `0`
     - :good:`100%`
   * - clang-tools-extra/clangd/tool
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - clang-tools-extra/clangd/unittests
     - `75`
     - `65`
     - `10`
     - :part:`86%`
   * - clang-tools-extra/clangd/unittests/decision_forest_model
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - clang-tools-extra/clangd/unittests/remote
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - clang-tools-extra/clangd/unittests/support
     - `10`
     - `10`
     - `0`
     - :good:`100%`
   * - clang-tools-extra/clangd/unittests/xpc
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - clang-tools-extra/clangd/xpc
     - `3`
     - `3`
     - `0`
     - :good:`100%`
   * - clang-tools-extra/clangd/xpc/framework
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - clang-tools-extra/clangd/xpc/test-client
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - clang-tools-extra/modularize
     - `9`
     - `1`
     - `8`
     - :part:`11%`
   * - clang-tools-extra/pp-trace
     - `3`
     - `1`
     - `2`
     - :part:`33%`
   * - clang-tools-extra/tool-template
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - clang-tools-extra/unittests/clang-apply-replacements
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - clang-tools-extra/unittests/clang-change-namespace
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - clang-tools-extra/unittests/clang-doc
     - `9`
     - `9`
     - `0`
     - :good:`100%`
   * - clang-tools-extra/unittests/clang-include-fixer
     - `2`
     - `0`
     - `2`
     - :none:`0%`
   * - clang-tools-extra/unittests/clang-include-fixer/find-all-symbols
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - clang-tools-extra/unittests/clang-move
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - clang-tools-extra/unittests/clang-query
     - `2`
     - `0`
     - `2`
     - :none:`0%`
   * - clang-tools-extra/unittests/clang-tidy
     - `15`
     - `7`
     - `8`
     - :part:`46%`
   * - clang-tools-extra/unittests/include/common
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - compiler-rt/include/fuzzer
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - compiler-rt/include/sanitizer
     - `15`
     - `2`
     - `13`
     - :part:`13%`
   * - compiler-rt/include/xray
     - `3`
     - `2`
     - `1`
     - :part:`66%`
   * - compiler-rt/lib/asan
     - `59`
     - `3`
     - `56`
     - :part:`5%`
   * - compiler-rt/lib/asan/tests
     - `17`
     - `1`
     - `16`
     - :part:`5%`
   * - compiler-rt/lib/BlocksRuntime
     - `2`
     - `0`
     - `2`
     - :none:`0%`
   * - compiler-rt/lib/builtins
     - `11`
     - `9`
     - `2`
     - :part:`81%`
   * - compiler-rt/lib/builtins/arm
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - compiler-rt/lib/builtins/ppc
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - compiler-rt/lib/cfi
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - compiler-rt/lib/dfsan
     - `5`
     - `1`
     - `4`
     - :part:`20%`
   * - compiler-rt/lib/fuzzer
     - `45`
     - `7`
     - `38`
     - :part:`15%`
   * - compiler-rt/lib/fuzzer/afl
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - compiler-rt/lib/fuzzer/dataflow
     - `3`
     - `0`
     - `3`
     - :none:`0%`
   * - compiler-rt/lib/fuzzer/tests
     - `2`
     - `1`
     - `1`
     - :part:`50%`
   * - compiler-rt/lib/gwp_asan
     - `13`
     - `13`
     - `0`
     - :good:`100%`
   * - compiler-rt/lib/gwp_asan/optional
     - `9`
     - `9`
     - `0`
     - :good:`100%`
   * - compiler-rt/lib/gwp_asan/platform_specific
     - `13`
     - `13`
     - `0`
     - :good:`100%`
   * - compiler-rt/lib/gwp_asan/tests
     - `14`
     - `14`
     - `0`
     - :good:`100%`
   * - compiler-rt/lib/gwp_asan/tests/optional
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - compiler-rt/lib/hwasan
     - `27`
     - `7`
     - `20`
     - :part:`25%`
   * - compiler-rt/lib/interception
     - `8`
     - `1`
     - `7`
     - :part:`12%`
   * - compiler-rt/lib/interception/tests
     - `3`
     - `1`
     - `2`
     - :part:`33%`
   * - compiler-rt/lib/lsan
     - `20`
     - `7`
     - `13`
     - :part:`35%`
   * - compiler-rt/lib/memprof
     - `27`
     - `27`
     - `0`
     - :good:`100%`
   * - compiler-rt/lib/msan
     - `18`
     - `4`
     - `14`
     - :part:`22%`
   * - compiler-rt/lib/msan/tests
     - `4`
     - `0`
     - `4`
     - :none:`0%`
   * - compiler-rt/lib/profile
     - `6`
     - `0`
     - `6`
     - :none:`0%`
   * - compiler-rt/lib/safestack
     - `3`
     - `1`
     - `2`
     - :part:`33%`
   * - compiler-rt/lib/sanitizer_common
     - `160`
     - `29`
     - `131`
     - :part:`18%`
   * - compiler-rt/lib/sanitizer_common/symbolizer
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - compiler-rt/lib/sanitizer_common/tests
     - `38`
     - `2`
     - `36`
     - :part:`5%`
   * - compiler-rt/lib/scudo
     - `20`
     - `0`
     - `20`
     - :none:`0%`
   * - compiler-rt/lib/scudo/standalone
     - `47`
     - `44`
     - `3`
     - :part:`93%`
   * - compiler-rt/lib/scudo/standalone/benchmarks
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - compiler-rt/lib/scudo/standalone/fuzz
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - compiler-rt/lib/scudo/standalone/include/scudo
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - compiler-rt/lib/scudo/standalone/tests
     - `23`
     - `23`
     - `0`
     - :good:`100%`
   * - compiler-rt/lib/scudo/standalone/tools
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - compiler-rt/lib/stats
     - `3`
     - `0`
     - `3`
     - :none:`0%`
   * - compiler-rt/lib/tsan/benchmarks
     - `6`
     - `0`
     - `6`
     - :none:`0%`
   * - compiler-rt/lib/tsan/dd
     - `3`
     - `0`
     - `3`
     - :none:`0%`
   * - compiler-rt/lib/tsan/go
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - compiler-rt/lib/tsan/rtl
     - `62`
     - `10`
     - `52`
     - :part:`16%`
   * - compiler-rt/lib/tsan/tests/rtl
     - `10`
     - `1`
     - `9`
     - :part:`10%`
   * - compiler-rt/lib/tsan/tests/unit
     - `10`
     - `0`
     - `10`
     - :none:`0%`
   * - compiler-rt/lib/ubsan
     - `27`
     - `7`
     - `20`
     - :part:`25%`
   * - compiler-rt/lib/ubsan_minimal
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - compiler-rt/lib/xray
     - `39`
     - `30`
     - `9`
     - :part:`76%`
   * - compiler-rt/lib/xray/tests/unit
     - `10`
     - `8`
     - `2`
     - :part:`80%`
   * - compiler-rt/tools/gwp_asan
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - debuginfo-tests/dexter/feature_tests/commands/penalty
     - `6`
     - `0`
     - `6`
     - :none:`0%`
   * - debuginfo-tests/dexter/feature_tests/commands/perfect
     - `5`
     - `0`
     - `5`
     - :none:`0%`
   * - debuginfo-tests/dexter/feature_tests/commands/perfect/expect_step_kind
     - `5`
     - `0`
     - `5`
     - :none:`0%`
   * - debuginfo-tests/dexter/feature_tests/commands/perfect/limit_steps
     - `5`
     - `0`
     - `5`
     - :none:`0%`
   * - debuginfo-tests/dexter/feature_tests/subtools
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - debuginfo-tests/dexter/feature_tests/subtools/clang-opt-bisect
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - debuginfo-tests/dexter-tests
     - `11`
     - `3`
     - `8`
     - :part:`27%`
   * - debuginfo-tests/llgdb-tests
     - `7`
     - `0`
     - `7`
     - :none:`0%`
   * - debuginfo-tests/llvm-prettyprinters/gdb
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - flang/include/flang
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - flang/include/flang/Common
     - `19`
     - `17`
     - `2`
     - :part:`89%`
   * - flang/include/flang/Decimal
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - flang/include/flang/Evaluate
     - `23`
     - `20`
     - `3`
     - :part:`86%`
   * - flang/include/flang/Frontend
     - `8`
     - `8`
     - `0`
     - :good:`100%`
   * - flang/include/flang/FrontendTool
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - flang/include/flang/Lower
     - `18`
     - `18`
     - `0`
     - :good:`100%`
   * - flang/include/flang/Lower/Support
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - flang/include/flang/Optimizer/CodeGen
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - flang/include/flang/Optimizer/Dialect
     - `5`
     - `3`
     - `2`
     - :part:`60%`
   * - flang/include/flang/Optimizer/Support
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - flang/include/flang/Optimizer/Transforms
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - flang/include/flang/Parser
     - `17`
     - `13`
     - `4`
     - :part:`76%`
   * - flang/include/flang/Semantics
     - `8`
     - `5`
     - `3`
     - :part:`62%`
   * - flang/lib/Common
     - `4`
     - `4`
     - `0`
     - :good:`100%`
   * - flang/lib/Decimal
     - `3`
     - `3`
     - `0`
     - :good:`100%`
   * - flang/lib/Evaluate
     - `31`
     - `29`
     - `2`
     - :part:`93%`
   * - flang/lib/Frontend
     - `8`
     - `7`
     - `1`
     - :part:`87%`
   * - flang/lib/FrontendTool
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - flang/lib/Lower
     - `16`
     - `16`
     - `0`
     - :good:`100%`
   * - flang/lib/Optimizer/Dialect
     - `4`
     - `2`
     - `2`
     - :part:`50%`
   * - flang/lib/Optimizer/Support
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - flang/lib/Optimizer/Transforms
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - flang/lib/Parser
     - `35`
     - `33`
     - `2`
     - :part:`94%`
   * - flang/lib/Semantics
     - `77`
     - `69`
     - `8`
     - :part:`89%`
   * - flang/module
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - flang/runtime
     - `59`
     - `54`
     - `5`
     - :part:`91%`
   * - flang/tools/f18
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - flang/tools/f18-parse-demo
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - flang/tools/flang-driver
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - flang/tools/tco
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - flang/unittests/Decimal
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - flang/unittests/Evaluate
     - `15`
     - `14`
     - `1`
     - :part:`93%`
   * - flang/unittests/Frontend
     - `3`
     - `3`
     - `0`
     - :good:`100%`
   * - flang/unittests/Optimizer
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - flang/unittests/Runtime
     - `8`
     - `8`
     - `0`
     - :good:`100%`
   * - libc/AOR_v20.02/math
     - `4`
     - `1`
     - `3`
     - :part:`25%`
   * - libc/AOR_v20.02/math/include
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - libc/AOR_v20.02/networking
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - libc/AOR_v20.02/networking/include
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - libc/AOR_v20.02/string
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - libc/AOR_v20.02/string/include
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - libc/benchmarks
     - `16`
     - `16`
     - `0`
     - :good:`100%`
   * - libc/config/linux
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - libc/fuzzing/math
     - `5`
     - `5`
     - `0`
     - :good:`100%`
   * - libc/fuzzing/string
     - `3`
     - `2`
     - `1`
     - :part:`66%`
   * - libc/include
     - `3`
     - `3`
     - `0`
     - :good:`100%`
   * - libc/loader/linux/x86_64
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - libc/src/assert
     - `3`
     - `1`
     - `2`
     - :part:`33%`
   * - libc/src/ctype
     - `29`
     - `29`
     - `0`
     - :good:`100%`
   * - libc/src/errno
     - `3`
     - `3`
     - `0`
     - :good:`100%`
   * - libc/src/fenv
     - `10`
     - `10`
     - `0`
     - :good:`100%`
   * - libc/src/math
     - `122`
     - `122`
     - `0`
     - :good:`100%`
   * - libc/src/signal
     - `8`
     - `8`
     - `0`
     - :good:`100%`
   * - libc/src/signal/linux
     - `10`
     - `10`
     - `0`
     - :good:`100%`
   * - libc/src/stdio
     - `3`
     - `3`
     - `0`
     - :good:`100%`
   * - libc/src/stdlib
     - `3`
     - `3`
     - `0`
     - :good:`100%`
   * - libc/src/stdlib/linux
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - libc/src/string
     - `39`
     - `39`
     - `0`
     - :good:`100%`
   * - libc/src/string/memory_utils
     - `3`
     - `3`
     - `0`
     - :good:`100%`
   * - libc/src/string/x86
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - libc/src/sys/mman
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - libc/src/sys/mman/linux
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - libc/src/threads
     - `6`
     - `6`
     - `0`
     - :good:`100%`
   * - libc/src/threads/linux
     - `7`
     - `7`
     - `0`
     - :good:`100%`
   * - libc/src/time
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - libc/src/unistd
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - libc/src/unistd/linux
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - libc/utils/CPP
     - `5`
     - `5`
     - `0`
     - :good:`100%`
   * - libc/utils/FPUtil
     - `18`
     - `18`
     - `0`
     - :good:`100%`
   * - libc/utils/FPUtil/x86_64
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - libc/utils/HdrGen
     - `9`
     - `9`
     - `0`
     - :good:`100%`
   * - libc/utils/HdrGen/PrototypeTestGen
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - libc/utils/LibcTableGenUtil
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - libc/utils/MPFRWrapper
     - `3`
     - `2`
     - `1`
     - :part:`66%`
   * - libc/utils/testutils
     - `6`
     - `6`
     - `0`
     - :good:`100%`
   * - libc/utils/tools/WrapperGen
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - libc/utils/UnitTest
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - libclc/generic/include
     - `2`
     - `1`
     - `1`
     - :part:`50%`
   * - libclc/generic/include/clc
     - `6`
     - `2`
     - `4`
     - :part:`33%`
   * - libclc/generic/include/clc/async
     - `4`
     - `4`
     - `0`
     - :good:`100%`
   * - libclc/generic/include/clc/atomic
     - `11`
     - `7`
     - `4`
     - :part:`63%`
   * - libclc/generic/include/clc/cl_khr_global_int32_base_atomics
     - `6`
     - `5`
     - `1`
     - :part:`83%`
   * - libclc/generic/include/clc/cl_khr_global_int32_extended_atomics
     - `5`
     - `5`
     - `0`
     - :good:`100%`
   * - libclc/generic/include/clc/cl_khr_int64_base_atomics
     - `6`
     - `3`
     - `3`
     - :part:`50%`
   * - libclc/generic/include/clc/cl_khr_int64_extended_atomics
     - `5`
     - `5`
     - `0`
     - :good:`100%`
   * - libclc/generic/include/clc/cl_khr_local_int32_base_atomics
     - `6`
     - `5`
     - `1`
     - :part:`83%`
   * - libclc/generic/include/clc/cl_khr_local_int32_extended_atomics
     - `5`
     - `5`
     - `0`
     - :good:`100%`
   * - libclc/generic/include/clc/common
     - `6`
     - `6`
     - `0`
     - :good:`100%`
   * - libclc/generic/include/clc/explicit_fence
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - libclc/generic/include/clc/float
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - libclc/generic/include/clc/geometric
     - `8`
     - `8`
     - `0`
     - :good:`100%`
   * - libclc/generic/include/clc/image
     - `2`
     - `0`
     - `2`
     - :none:`0%`
   * - libclc/generic/include/clc/integer
     - `16`
     - `13`
     - `3`
     - :part:`81%`
   * - libclc/generic/include/clc/math
     - `95`
     - `92`
     - `3`
     - :part:`96%`
   * - libclc/generic/include/clc/misc
     - `2`
     - `0`
     - `2`
     - :none:`0%`
   * - libclc/generic/include/clc/relational
     - `18`
     - `12`
     - `6`
     - :part:`66%`
   * - libclc/generic/include/clc/shared
     - `5`
     - `3`
     - `2`
     - :part:`60%`
   * - libclc/generic/include/clc/synchronization
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - libclc/generic/include/clc/workitem
     - `8`
     - `8`
     - `0`
     - :good:`100%`
   * - libclc/generic/include/integer
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - libclc/generic/include/math
     - `15`
     - `15`
     - `0`
     - :good:`100%`
   * - libclc/generic/lib
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - libclc/generic/lib/math
     - `8`
     - `1`
     - `7`
     - :part:`12%`
   * - libclc/generic/lib/relational
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - libclc/utils
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - libcxx/benchmarks
     - `21`
     - `6`
     - `15`
     - :part:`28%`
   * - libcxx/include
     - `21`
     - `0`
     - `21`
     - :none:`0%`
   * - libcxx/include/support/android
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - libcxx/include/support/fuchsia
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - libcxx/include/support/ibm
     - `5`
     - `1`
     - `4`
     - :part:`20%`
   * - libcxx/include/support/musl
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - libcxx/include/support/newlib
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - libcxx/include/support/nuttx
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - libcxx/include/support/solaris
     - `3`
     - `2`
     - `1`
     - :part:`66%`
   * - libcxx/include/support/win32
     - `2`
     - `0`
     - `2`
     - :none:`0%`
   * - libcxx/include/support/xlocale
     - `3`
     - `0`
     - `3`
     - :none:`0%`
   * - libcxx/src
     - `37`
     - `1`
     - `36`
     - :part:`2%`
   * - libcxx/src/experimental
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - libcxx/src/filesystem
     - `4`
     - `1`
     - `3`
     - :part:`25%`
   * - libcxx/src/include
     - `4`
     - `2`
     - `2`
     - :part:`50%`
   * - libcxx/src/support/solaris
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - libcxx/src/support/win32
     - `3`
     - `0`
     - `3`
     - :none:`0%`
   * - libcxx/utils/google-benchmark/cmake
     - `5`
     - `1`
     - `4`
     - :part:`20%`
   * - libcxx/utils/google-benchmark/include/benchmark
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - libcxx/utils/google-benchmark/src
     - `20`
     - `16`
     - `4`
     - :part:`80%`
   * - libcxxabi/fuzz
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - libcxxabi/include
     - `2`
     - `0`
     - `2`
     - :none:`0%`
   * - libcxxabi/src
     - `25`
     - `0`
     - `25`
     - :none:`0%`
   * - libcxxabi/src/demangle
     - `4`
     - `2`
     - `2`
     - :part:`50%`
   * - libcxxabi/src/include
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - libunwind/include
     - `3`
     - `0`
     - `3`
     - :none:`0%`
   * - libunwind/include/mach-o
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - libunwind/src
     - `9`
     - `0`
     - `9`
     - :none:`0%`
   * - lld/COFF
     - `35`
     - `11`
     - `24`
     - :part:`31%`
   * - lld/Common
     - `10`
     - `8`
     - `2`
     - :part:`80%`
   * - lld/ELF
     - `48`
     - `25`
     - `23`
     - :part:`52%`
   * - lld/ELF/Arch
     - `14`
     - `7`
     - `7`
     - :part:`50%`
   * - lld/include/lld/Common
     - `12`
     - `6`
     - `6`
     - :part:`50%`
   * - lld/include/lld/Core
     - `20`
     - `4`
     - `16`
     - :part:`20%`
   * - lld/include/lld/ReaderWriter
     - `2`
     - `0`
     - `2`
     - :none:`0%`
   * - lld/lib/Core
     - `8`
     - `2`
     - `6`
     - :part:`25%`
   * - lld/lib/Driver
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - lld/lib/ReaderWriter
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - lld/lib/ReaderWriter/MachO
     - `30`
     - `1`
     - `29`
     - :part:`3%`
   * - lld/lib/ReaderWriter/YAML
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - lld/MachO
     - `35`
     - `31`
     - `4`
     - :part:`88%`
   * - lld/MachO/Arch
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - lld/MinGW
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - lld/tools/lld
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - lld/unittests/DriverTests
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - lld/unittests/MachOTests
     - `4`
     - `0`
     - `4`
     - :none:`0%`
   * - lld/wasm
     - `29`
     - `16`
     - `13`
     - :part:`55%`
   * - lldb/bindings/python
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - lldb/examples/darwin/heap_find/heap
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - lldb/examples/functions
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - lldb/examples/interposing/darwin/fd_interposing
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - lldb/examples/lookup
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - lldb/examples/plugins/commands
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - lldb/examples/synthetic/bitfield
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - lldb/include/lldb
     - `12`
     - `6`
     - `6`
     - :part:`50%`
   * - lldb/include/lldb/API
     - `71`
     - `59`
     - `12`
     - :part:`83%`
   * - lldb/include/lldb/Breakpoint
     - `25`
     - `9`
     - `16`
     - :part:`36%`
   * - lldb/include/lldb/Core
     - `57`
     - `30`
     - `27`
     - :part:`52%`
   * - lldb/include/lldb/DataFormatters
     - `18`
     - `10`
     - `8`
     - :part:`55%`
   * - lldb/include/lldb/Expression
     - `17`
     - `7`
     - `10`
     - :part:`41%`
   * - lldb/include/lldb/Host
     - `40`
     - `20`
     - `20`
     - :part:`50%`
   * - lldb/include/lldb/Host/android
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - lldb/include/lldb/Host/common
     - `8`
     - `2`
     - `6`
     - :part:`25%`
   * - lldb/include/lldb/Host/freebsd
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - lldb/include/lldb/Host/linux
     - `5`
     - `3`
     - `2`
     - :part:`60%`
   * - lldb/include/lldb/Host/macosx
     - `2`
     - `0`
     - `2`
     - :none:`0%`
   * - lldb/include/lldb/Host/netbsd
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - lldb/include/lldb/Host/openbsd
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - lldb/include/lldb/Host/posix
     - `9`
     - `7`
     - `2`
     - :part:`77%`
   * - lldb/include/lldb/Host/windows
     - `11`
     - `5`
     - `6`
     - :part:`45%`
   * - lldb/include/lldb/Initialization
     - `3`
     - `1`
     - `2`
     - :part:`33%`
   * - lldb/include/lldb/Interpreter
     - `47`
     - `39`
     - `8`
     - :part:`82%`
   * - lldb/include/lldb/Symbol
     - `36`
     - `17`
     - `19`
     - :part:`47%`
   * - lldb/include/lldb/Target
     - `72`
     - `43`
     - `29`
     - :part:`59%`
   * - lldb/include/lldb/Utility
     - `61`
     - `40`
     - `21`
     - :part:`65%`
   * - lldb/source
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - lldb/source/API
     - `75`
     - `7`
     - `68`
     - :part:`9%`
   * - lldb/source/Breakpoint
     - `24`
     - `5`
     - `19`
     - :part:`20%`
   * - lldb/source/Commands
     - `66`
     - `58`
     - `8`
     - :part:`87%`
   * - lldb/source/Core
     - `45`
     - `24`
     - `21`
     - :part:`53%`
   * - lldb/source/DataFormatters
     - `16`
     - `3`
     - `13`
     - :part:`18%`
   * - lldb/source/Expression
     - `13`
     - `4`
     - `9`
     - :part:`30%`
   * - lldb/source/Host/android
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - lldb/source/Host/common
     - `32`
     - `17`
     - `15`
     - :part:`53%`
   * - lldb/source/Host/freebsd
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - lldb/source/Host/linux
     - `5`
     - `4`
     - `1`
     - :part:`80%`
   * - lldb/source/Host/macosx/cfcpp
     - `14`
     - `12`
     - `2`
     - :part:`85%`
   * - lldb/source/Host/macosx/objcxx
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - lldb/source/Host/netbsd
     - `2`
     - `0`
     - `2`
     - :none:`0%`
   * - lldb/source/Host/openbsd
     - `2`
     - `1`
     - `1`
     - :part:`50%`
   * - lldb/source/Host/posix
     - `9`
     - `5`
     - `4`
     - :part:`55%`
   * - lldb/source/Host/windows
     - `12`
     - `5`
     - `7`
     - :part:`41%`
   * - lldb/source/Initialization
     - `3`
     - `3`
     - `0`
     - :good:`100%`
   * - lldb/source/Interpreter
     - `44`
     - `23`
     - `21`
     - :part:`52%`
   * - lldb/source/Plugins/ABI/AArch64
     - `6`
     - `2`
     - `4`
     - :part:`33%`
   * - lldb/source/Plugins/ABI/ARC
     - `2`
     - `0`
     - `2`
     - :none:`0%`
   * - lldb/source/Plugins/ABI/ARM
     - `6`
     - `4`
     - `2`
     - :part:`66%`
   * - lldb/source/Plugins/ABI/Hexagon
     - `2`
     - `0`
     - `2`
     - :none:`0%`
   * - lldb/source/Plugins/ABI/Mips
     - `6`
     - `2`
     - `4`
     - :part:`33%`
   * - lldb/source/Plugins/ABI/PowerPC
     - `6`
     - `3`
     - `3`
     - :part:`50%`
   * - lldb/source/Plugins/ABI/SystemZ
     - `2`
     - `0`
     - `2`
     - :none:`0%`
   * - lldb/source/Plugins/ABI/X86
     - `11`
     - `4`
     - `7`
     - :part:`36%`
   * - lldb/source/Plugins/Architecture/Arm
     - `2`
     - `1`
     - `1`
     - :part:`50%`
   * - lldb/source/Plugins/Architecture/Mips
     - `2`
     - `0`
     - `2`
     - :none:`0%`
   * - lldb/source/Plugins/Architecture/PPC64
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - lldb/source/Plugins/Disassembler/LLVMC
     - `2`
     - `1`
     - `1`
     - :part:`50%`
   * - lldb/source/Plugins/DynamicLoader/Darwin-Kernel
     - `2`
     - `0`
     - `2`
     - :none:`0%`
   * - lldb/source/Plugins/DynamicLoader/Hexagon-DYLD
     - `4`
     - `4`
     - `0`
     - :good:`100%`
   * - lldb/source/Plugins/DynamicLoader/MacOSX-DYLD
     - `6`
     - `3`
     - `3`
     - :part:`50%`
   * - lldb/source/Plugins/DynamicLoader/POSIX-DYLD
     - `4`
     - `2`
     - `2`
     - :part:`50%`
   * - lldb/source/Plugins/DynamicLoader/Static
     - `2`
     - `1`
     - `1`
     - :part:`50%`
   * - lldb/source/Plugins/DynamicLoader/wasm-DYLD
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - lldb/source/Plugins/DynamicLoader/Windows-DYLD
     - `2`
     - `1`
     - `1`
     - :part:`50%`
   * - lldb/source/Plugins/ExpressionParser/Clang
     - `51`
     - `26`
     - `25`
     - :part:`50%`
   * - lldb/source/Plugins/Instruction/ARM
     - `4`
     - `2`
     - `2`
     - :part:`50%`
   * - lldb/source/Plugins/Instruction/ARM64
     - `2`
     - `0`
     - `2`
     - :none:`0%`
   * - lldb/source/Plugins/Instruction/MIPS
     - `2`
     - `0`
     - `2`
     - :none:`0%`
   * - lldb/source/Plugins/Instruction/MIPS64
     - `2`
     - `1`
     - `1`
     - :part:`50%`
   * - lldb/source/Plugins/Instruction/PPC64
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - lldb/source/Plugins/InstrumentationRuntime/ASan
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - lldb/source/Plugins/InstrumentationRuntime/MainThreadChecker
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - lldb/source/Plugins/InstrumentationRuntime/TSan
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - lldb/source/Plugins/InstrumentationRuntime/UBSan
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - lldb/source/Plugins/JITLoader/GDB
     - `2`
     - `1`
     - `1`
     - :part:`50%`
   * - lldb/source/Plugins/Language/ClangCommon
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - lldb/source/Plugins/Language/CPlusPlus
     - `29`
     - `17`
     - `12`
     - :part:`58%`
   * - lldb/source/Plugins/Language/ObjC
     - `20`
     - `13`
     - `7`
     - :part:`65%`
   * - lldb/source/Plugins/Language/ObjCPlusPlus
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - lldb/source/Plugins/LanguageRuntime/CPlusPlus
     - `2`
     - `0`
     - `2`
     - :none:`0%`
   * - lldb/source/Plugins/LanguageRuntime/CPlusPlus/ItaniumABI
     - `2`
     - `0`
     - `2`
     - :none:`0%`
   * - lldb/source/Plugins/LanguageRuntime/ObjC
     - `2`
     - `1`
     - `1`
     - :part:`50%`
   * - lldb/source/Plugins/LanguageRuntime/ObjC/AppleObjCRuntime
     - `16`
     - `4`
     - `12`
     - :part:`25%`
   * - lldb/source/Plugins/LanguageRuntime/RenderScript/RenderScriptRuntime
     - `8`
     - `3`
     - `5`
     - :part:`37%`
   * - lldb/source/Plugins/MemoryHistory/asan
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - lldb/source/Plugins/ObjectContainer/BSD-Archive
     - `2`
     - `0`
     - `2`
     - :none:`0%`
   * - lldb/source/Plugins/ObjectContainer/Universal-Mach-O
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - lldb/source/Plugins/ObjectFile/Breakpad
     - `4`
     - `3`
     - `1`
     - :part:`75%`
   * - lldb/source/Plugins/ObjectFile/ELF
     - `4`
     - `1`
     - `3`
     - :part:`25%`
   * - lldb/source/Plugins/ObjectFile/JIT
     - `2`
     - `0`
     - `2`
     - :none:`0%`
   * - lldb/source/Plugins/ObjectFile/Mach-O
     - `2`
     - `0`
     - `2`
     - :none:`0%`
   * - lldb/source/Plugins/ObjectFile/PDB
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - lldb/source/Plugins/ObjectFile/PECOFF
     - `6`
     - `3`
     - `3`
     - :part:`50%`
   * - lldb/source/Plugins/ObjectFile/wasm
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - lldb/source/Plugins/OperatingSystem/Python
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - lldb/source/Plugins/Platform/Android
     - `6`
     - `3`
     - `3`
     - :part:`50%`
   * - lldb/source/Plugins/Platform/FreeBSD
     - `2`
     - `1`
     - `1`
     - :part:`50%`
   * - lldb/source/Plugins/Platform/gdb-server
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - lldb/source/Plugins/Platform/Linux
     - `2`
     - `1`
     - `1`
     - :part:`50%`
   * - lldb/source/Plugins/Platform/MacOSX
     - `20`
     - `11`
     - `9`
     - :part:`55%`
   * - lldb/source/Plugins/Platform/MacOSX/objcxx
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - lldb/source/Plugins/Platform/NetBSD
     - `2`
     - `1`
     - `1`
     - :part:`50%`
   * - lldb/source/Plugins/Platform/OpenBSD
     - `2`
     - `1`
     - `1`
     - :part:`50%`
   * - lldb/source/Plugins/Platform/POSIX
     - `2`
     - `0`
     - `2`
     - :none:`0%`
   * - lldb/source/Plugins/Platform/Windows
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - lldb/source/Plugins/Process/elf-core
     - `20`
     - `18`
     - `2`
     - :part:`90%`
   * - lldb/source/Plugins/Process/FreeBSD
     - `19`
     - `11`
     - `8`
     - :part:`57%`
   * - lldb/source/Plugins/Process/FreeBSDRemote
     - `8`
     - `8`
     - `0`
     - :good:`100%`
   * - lldb/source/Plugins/Process/gdb-remote
     - `26`
     - `16`
     - `10`
     - :part:`61%`
   * - lldb/source/Plugins/Process/Linux
     - `23`
     - `11`
     - `12`
     - :part:`47%`
   * - lldb/source/Plugins/Process/mach-core
     - `4`
     - `3`
     - `1`
     - :part:`75%`
   * - lldb/source/Plugins/Process/MacOSX-Kernel
     - `16`
     - `13`
     - `3`
     - :part:`81%`
   * - lldb/source/Plugins/Process/minidump
     - `17`
     - `10`
     - `7`
     - :part:`58%`
   * - lldb/source/Plugins/Process/NetBSD
     - `8`
     - `4`
     - `4`
     - :part:`50%`
   * - lldb/source/Plugins/Process/POSIX
     - `8`
     - `5`
     - `3`
     - :part:`62%`
   * - lldb/source/Plugins/Process/Utility
     - `133`
     - `92`
     - `41`
     - :part:`69%`
   * - lldb/source/Plugins/Process/Windows/Common
     - `34`
     - `23`
     - `11`
     - :part:`67%`
   * - lldb/source/Plugins/Process/Windows/Common/arm
     - `2`
     - `1`
     - `1`
     - :part:`50%`
   * - lldb/source/Plugins/Process/Windows/Common/arm64
     - `2`
     - `1`
     - `1`
     - :part:`50%`
   * - lldb/source/Plugins/Process/Windows/Common/x64
     - `2`
     - `0`
     - `2`
     - :none:`0%`
   * - lldb/source/Plugins/Process/Windows/Common/x86
     - `2`
     - `0`
     - `2`
     - :none:`0%`
   * - lldb/source/Plugins/ScriptInterpreter/Lua
     - `4`
     - `3`
     - `1`
     - :part:`75%`
   * - lldb/source/Plugins/ScriptInterpreter/None
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - lldb/source/Plugins/ScriptInterpreter/Python
     - `8`
     - `3`
     - `5`
     - :part:`37%`
   * - lldb/source/Plugins/StructuredData/DarwinLog
     - `2`
     - `0`
     - `2`
     - :none:`0%`
   * - lldb/source/Plugins/SymbolFile/Breakpad
     - `2`
     - `0`
     - `2`
     - :none:`0%`
   * - lldb/source/Plugins/SymbolFile/DWARF
     - `65`
     - `36`
     - `29`
     - :part:`55%`
   * - lldb/source/Plugins/SymbolFile/NativePDB
     - `20`
     - `11`
     - `9`
     - :part:`55%`
   * - lldb/source/Plugins/SymbolFile/PDB
     - `6`
     - `4`
     - `2`
     - :part:`66%`
   * - lldb/source/Plugins/SymbolFile/Symtab
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - lldb/source/Plugins/SymbolVendor/ELF
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - lldb/source/Plugins/SymbolVendor/MacOSX
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - lldb/source/Plugins/SymbolVendor/wasm
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - lldb/source/Plugins/SystemRuntime/MacOSX
     - `10`
     - `1`
     - `9`
     - :part:`10%`
   * - lldb/source/Plugins/Trace/intel-pt
     - `10`
     - `10`
     - `0`
     - :good:`100%`
   * - lldb/source/Plugins/TypeSystem/Clang
     - `2`
     - `0`
     - `2`
     - :none:`0%`
   * - lldb/source/Plugins/UnwindAssembly/InstEmulation
     - `2`
     - `1`
     - `1`
     - :part:`50%`
   * - lldb/source/Plugins/UnwindAssembly/x86
     - `4`
     - `2`
     - `2`
     - :part:`50%`
   * - lldb/source/Symbol
     - `32`
     - `18`
     - `14`
     - :part:`56%`
   * - lldb/source/Target
     - `65`
     - `32`
     - `33`
     - :part:`49%`
   * - lldb/source/Utility
     - `57`
     - `44`
     - `13`
     - :part:`77%`
   * - lldb/tools/argdumper
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - lldb/tools/darwin-debug
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - lldb/tools/debugserver/source
     - `49`
     - `39`
     - `10`
     - :part:`79%`
   * - lldb/tools/debugserver/source/MacOSX
     - `24`
     - `16`
     - `8`
     - :part:`66%`
   * - lldb/tools/debugserver/source/MacOSX/arm
     - `2`
     - `1`
     - `1`
     - :part:`50%`
   * - lldb/tools/debugserver/source/MacOSX/arm64
     - `2`
     - `1`
     - `1`
     - :part:`50%`
   * - lldb/tools/debugserver/source/MacOSX/DarwinLog
     - `20`
     - `18`
     - `2`
     - :part:`90%`
   * - lldb/tools/debugserver/source/MacOSX/i386
     - `3`
     - `0`
     - `3`
     - :none:`0%`
   * - lldb/tools/debugserver/source/MacOSX/x86_64
     - `3`
     - `0`
     - `3`
     - :none:`0%`
   * - lldb/tools/driver
     - `4`
     - `4`
     - `0`
     - :good:`100%`
   * - lldb/tools/intel-features
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - lldb/tools/intel-features/intel-mpx
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - lldb/tools/intel-features/intel-pt
     - `6`
     - `6`
     - `0`
     - :good:`100%`
   * - lldb/tools/lldb-instr
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - lldb/tools/lldb-server
     - `9`
     - `4`
     - `5`
     - :part:`44%`
   * - lldb/tools/lldb-test
     - `5`
     - `3`
     - `2`
     - :part:`60%`
   * - lldb/tools/lldb-vscode
     - `19`
     - `18`
     - `1`
     - :part:`94%`
   * - lldb/unittests
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - lldb/unittests/API
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - lldb/unittests/Breakpoint
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - lldb/unittests/Core
     - `7`
     - `6`
     - `1`
     - :part:`85%`
   * - lldb/unittests/DataFormatter
     - `3`
     - `3`
     - `0`
     - :good:`100%`
   * - lldb/unittests/debugserver
     - `3`
     - `2`
     - `1`
     - :part:`66%`
   * - lldb/unittests/Disassembler
     - `2`
     - `0`
     - `2`
     - :none:`0%`
   * - lldb/unittests/Editline
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - lldb/unittests/Expression
     - `5`
     - `3`
     - `2`
     - :part:`60%`
   * - lldb/unittests/Host
     - `13`
     - `10`
     - `3`
     - :part:`76%`
   * - lldb/unittests/Host/linux
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - lldb/unittests/Instruction
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - lldb/unittests/Interpreter
     - `3`
     - `1`
     - `2`
     - :part:`33%`
   * - lldb/unittests/Language/CLanguages
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - lldb/unittests/Language/CPlusPlus
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - lldb/unittests/Language/Highlighting
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - lldb/unittests/ObjectFile/Breakpad
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - lldb/unittests/ObjectFile/ELF
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - lldb/unittests/ObjectFile/MachO
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - lldb/unittests/ObjectFile/PECOFF
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - lldb/unittests/Platform
     - `2`
     - `1`
     - `1`
     - :part:`50%`
   * - lldb/unittests/Platform/Android
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - lldb/unittests/Process
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - lldb/unittests/Process/gdb-remote
     - `7`
     - `6`
     - `1`
     - :part:`85%`
   * - lldb/unittests/Process/Linux
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - lldb/unittests/Process/minidump
     - `2`
     - `0`
     - `2`
     - :none:`0%`
   * - lldb/unittests/Process/minidump/Inputs
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - lldb/unittests/Process/POSIX
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - lldb/unittests/Process/Utility
     - `3`
     - `3`
     - `0`
     - :good:`100%`
   * - lldb/unittests/ScriptInterpreter/Lua
     - `2`
     - `1`
     - `1`
     - :part:`50%`
   * - lldb/unittests/ScriptInterpreter/Python
     - `3`
     - `1`
     - `2`
     - :part:`33%`
   * - lldb/unittests/Signals
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - lldb/unittests/Symbol
     - `7`
     - `4`
     - `3`
     - :part:`57%`
   * - lldb/unittests/SymbolFile/DWARF
     - `3`
     - `0`
     - `3`
     - :none:`0%`
   * - lldb/unittests/SymbolFile/DWARF/Inputs
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - lldb/unittests/SymbolFile/NativePDB
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - lldb/unittests/SymbolFile/PDB
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - lldb/unittests/SymbolFile/PDB/Inputs
     - `5`
     - `5`
     - `0`
     - :good:`100%`
   * - lldb/unittests/Target
     - `7`
     - `3`
     - `4`
     - :part:`42%`
   * - lldb/unittests/TestingSupport
     - `5`
     - `4`
     - `1`
     - :part:`80%`
   * - lldb/unittests/TestingSupport/Host
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - lldb/unittests/TestingSupport/Symbol
     - `3`
     - `3`
     - `0`
     - :good:`100%`
   * - lldb/unittests/Thread
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - lldb/unittests/tools/lldb-server/inferior
     - `2`
     - `0`
     - `2`
     - :none:`0%`
   * - lldb/unittests/tools/lldb-server/tests
     - `8`
     - `1`
     - `7`
     - :part:`12%`
   * - lldb/unittests/UnwindAssembly/ARM64
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - lldb/unittests/UnwindAssembly/PPC64
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - lldb/unittests/UnwindAssembly/x86
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - lldb/unittests/Utility
     - `44`
     - `32`
     - `12`
     - :part:`72%`
   * - lldb/utils/lit-cpuid
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - lldb/utils/TableGen
     - `6`
     - `6`
     - `0`
     - :good:`100%`
   * - llvm/benchmarks
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/bindings/go/llvm
     - `6`
     - `3`
     - `3`
     - :part:`50%`
   * - llvm/cmake
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - llvm/examples/BrainF
     - `3`
     - `0`
     - `3`
     - :none:`0%`
   * - llvm/examples/Bye
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - llvm/examples/ExceptionDemo
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/examples/Fibonacci
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/examples/HowToUseJIT
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/examples/HowToUseLLJIT
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - llvm/examples/IRTransforms
     - `4`
     - `4`
     - `0`
     - :good:`100%`
   * - llvm/examples/Kaleidoscope/BuildingAJIT/Chapter1
     - `2`
     - `1`
     - `1`
     - :part:`50%`
   * - llvm/examples/Kaleidoscope/BuildingAJIT/Chapter2
     - `2`
     - `1`
     - `1`
     - :part:`50%`
   * - llvm/examples/Kaleidoscope/BuildingAJIT/Chapter3
     - `2`
     - `1`
     - `1`
     - :part:`50%`
   * - llvm/examples/Kaleidoscope/BuildingAJIT/Chapter4
     - `2`
     - `0`
     - `2`
     - :none:`0%`
   * - llvm/examples/Kaleidoscope/Chapter2
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - llvm/examples/Kaleidoscope/Chapter3
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/examples/Kaleidoscope/Chapter4
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/examples/Kaleidoscope/Chapter5
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/examples/Kaleidoscope/Chapter6
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/examples/Kaleidoscope/Chapter7
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/examples/Kaleidoscope/Chapter8
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/examples/Kaleidoscope/Chapter9
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/examples/Kaleidoscope/include
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - llvm/examples/Kaleidoscope/MCJIT/cached
     - `2`
     - `0`
     - `2`
     - :none:`0%`
   * - llvm/examples/Kaleidoscope/MCJIT/complete
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/examples/Kaleidoscope/MCJIT/initial
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/examples/Kaleidoscope/MCJIT/lazy
     - `2`
     - `0`
     - `2`
     - :none:`0%`
   * - llvm/examples/ModuleMaker
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/examples/OrcV2Examples
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - llvm/examples/OrcV2Examples/LLJITDumpObjects
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - llvm/examples/OrcV2Examples/LLJITWithCustomObjectLinkingLayer
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - llvm/examples/OrcV2Examples/LLJITWithGDBRegistrationListener
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - llvm/examples/OrcV2Examples/LLJITWithInitializers
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - llvm/examples/OrcV2Examples/LLJITWithLazyReexports
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - llvm/examples/OrcV2Examples/LLJITWithObjectCache
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - llvm/examples/OrcV2Examples/LLJITWithObjectLinkingLayerPlugin
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - llvm/examples/OrcV2Examples/LLJITWithOptimizingIRTransform
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - llvm/examples/OrcV2Examples/LLJITWithTargetProcessControl
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - llvm/examples/OrcV2Examples/LLJITWithThinLTOSummaries
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/examples/ParallelJIT
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/examples/SpeculativeJIT
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/include/llvm
     - `8`
     - `2`
     - `6`
     - :part:`25%`
   * - llvm/include/llvm/ADT
     - `86`
     - `25`
     - `61`
     - :part:`29%`
   * - llvm/include/llvm/Analysis
     - `120`
     - `40`
     - `80`
     - :part:`33%`
   * - llvm/include/llvm/Analysis/Utils
     - `2`
     - `1`
     - `1`
     - :part:`50%`
   * - llvm/include/llvm/AsmParser
     - `2`
     - `1`
     - `1`
     - :part:`50%`
   * - llvm/include/llvm/BinaryFormat
     - `14`
     - `9`
     - `5`
     - :part:`64%`
   * - llvm/include/llvm/Bitcode
     - `6`
     - `2`
     - `4`
     - :part:`33%`
   * - llvm/include/llvm/Bitstream
     - `3`
     - `0`
     - `3`
     - :none:`0%`
   * - llvm/include/llvm/CodeGen
     - `146`
     - `38`
     - `108`
     - :part:`26%`
   * - llvm/include/llvm/CodeGen/GlobalISel
     - `27`
     - `10`
     - `17`
     - :part:`37%`
   * - llvm/include/llvm/CodeGen/MIRParser
     - `2`
     - `1`
     - `1`
     - :part:`50%`
   * - llvm/include/llvm/CodeGen/PBQP
     - `5`
     - `1`
     - `4`
     - :part:`20%`
   * - llvm/include/llvm/DebugInfo
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/include/llvm/DebugInfo/CodeView
     - `57`
     - `40`
     - `17`
     - :part:`70%`
   * - llvm/include/llvm/DebugInfo/DWARF
     - `32`
     - `17`
     - `15`
     - :part:`53%`
   * - llvm/include/llvm/DebugInfo/GSYM
     - `14`
     - `2`
     - `12`
     - :part:`14%`
   * - llvm/include/llvm/DebugInfo/MSF
     - `5`
     - `4`
     - `1`
     - :part:`80%`
   * - llvm/include/llvm/DebugInfo/PDB
     - `50`
     - `7`
     - `43`
     - :part:`14%`
   * - llvm/include/llvm/DebugInfo/PDB/DIA
     - `20`
     - `9`
     - `11`
     - :part:`45%`
   * - llvm/include/llvm/DebugInfo/PDB/Native
     - `54`
     - `35`
     - `19`
     - :part:`64%`
   * - llvm/include/llvm/DebugInfo/Symbolize
     - `3`
     - `0`
     - `3`
     - :none:`0%`
   * - llvm/include/llvm/Demangle
     - `7`
     - `3`
     - `4`
     - :part:`42%`
   * - llvm/include/llvm/DWARFLinker
     - `4`
     - `4`
     - `0`
     - :good:`100%`
   * - llvm/include/llvm/ExecutionEngine
     - `14`
     - `3`
     - `11`
     - :part:`21%`
   * - llvm/include/llvm/ExecutionEngine/JITLink
     - `8`
     - `5`
     - `3`
     - :part:`62%`
   * - llvm/include/llvm/ExecutionEngine/Orc
     - `31`
     - `19`
     - `12`
     - :part:`61%`
   * - llvm/include/llvm/ExecutionEngine/Orc/RPC
     - `4`
     - `1`
     - `3`
     - :part:`25%`
   * - llvm/include/llvm/ExecutionEngine/Orc/Shared
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - llvm/include/llvm/ExecutionEngine/Orc/TargetProcess
     - `3`
     - `3`
     - `0`
     - :good:`100%`
   * - llvm/include/llvm/FileCheck
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - llvm/include/llvm/Frontend/OpenMP
     - `4`
     - `4`
     - `0`
     - :good:`100%`
   * - llvm/include/llvm/FuzzMutate
     - `6`
     - `0`
     - `6`
     - :none:`0%`
   * - llvm/include/llvm/InterfaceStub
     - `3`
     - `3`
     - `0`
     - :good:`100%`
   * - llvm/include/llvm/IR
     - `88`
     - `22`
     - `66`
     - :part:`25%`
   * - llvm/include/llvm/IRReader
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/include/llvm/LineEditor
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/include/llvm/Linker
     - `2`
     - `0`
     - `2`
     - :none:`0%`
   * - llvm/include/llvm/LTO
     - `5`
     - `2`
     - `3`
     - :part:`40%`
   * - llvm/include/llvm/LTO/legacy
     - `4`
     - `0`
     - `4`
     - :none:`0%`
   * - llvm/include/llvm/MC
     - `69`
     - `17`
     - `52`
     - :part:`24%`
   * - llvm/include/llvm/MC/MCDisassembler
     - `4`
     - `1`
     - `3`
     - :part:`25%`
   * - llvm/include/llvm/MC/MCParser
     - `8`
     - `3`
     - `5`
     - :part:`37%`
   * - llvm/include/llvm/MCA
     - `8`
     - `8`
     - `0`
     - :good:`100%`
   * - llvm/include/llvm/MCA/HardwareUnits
     - `6`
     - `4`
     - `2`
     - :part:`66%`
   * - llvm/include/llvm/MCA/Stages
     - `7`
     - `6`
     - `1`
     - :part:`85%`
   * - llvm/include/llvm/Object
     - `30`
     - `10`
     - `20`
     - :part:`33%`
   * - llvm/include/llvm/ObjectYAML
     - `16`
     - `13`
     - `3`
     - :part:`81%`
   * - llvm/include/llvm/Option
     - `5`
     - `1`
     - `4`
     - :part:`20%`
   * - llvm/include/llvm/Passes
     - `3`
     - `1`
     - `2`
     - :part:`33%`
   * - llvm/include/llvm/ProfileData
     - `8`
     - `4`
     - `4`
     - :part:`50%`
   * - llvm/include/llvm/ProfileData/Coverage
     - `3`
     - `2`
     - `1`
     - :part:`66%`
   * - llvm/include/llvm/Remarks
     - `12`
     - `11`
     - `1`
     - :part:`91%`
   * - llvm/include/llvm/Support
     - `171`
     - `53`
     - `118`
     - :part:`30%`
   * - llvm/include/llvm/Support/FileSystem
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - llvm/include/llvm/Support/Solaris/sys
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - llvm/include/llvm/Support/Windows
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/include/llvm/TableGen
     - `8`
     - `2`
     - `6`
     - :part:`25%`
   * - llvm/include/llvm/Target
     - `5`
     - `1`
     - `4`
     - :part:`20%`
   * - llvm/include/llvm/Testing/Support
     - `3`
     - `2`
     - `1`
     - :part:`66%`
   * - llvm/include/llvm/TextAPI
     - `9`
     - `8`
     - `1`
     - :part:`88%`
   * - llvm/include/llvm/ToolDrivers/llvm-dlltool
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - llvm/include/llvm/ToolDrivers/llvm-lib
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/include/llvm/Transforms
     - `8`
     - `2`
     - `6`
     - :part:`25%`
   * - llvm/include/llvm/Transforms/AggressiveInstCombine
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/include/llvm/Transforms/Coroutines
     - `4`
     - `4`
     - `0`
     - :good:`100%`
   * - llvm/include/llvm/Transforms/HelloNew
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - llvm/include/llvm/Transforms/InstCombine
     - `3`
     - `1`
     - `2`
     - :part:`33%`
   * - llvm/include/llvm/Transforms/Instrumentation
     - `16`
     - `10`
     - `6`
     - :part:`62%`
   * - llvm/include/llvm/Transforms/IPO
     - `34`
     - `23`
     - `11`
     - :part:`67%`
   * - llvm/include/llvm/Transforms/Scalar
     - `71`
     - `42`
     - `29`
     - :part:`59%`
   * - llvm/include/llvm/Transforms/Utils
     - `66`
     - `35`
     - `31`
     - :part:`53%`
   * - llvm/include/llvm/Transforms/Vectorize
     - `5`
     - `2`
     - `3`
     - :part:`40%`
   * - llvm/include/llvm/WindowsManifest
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - llvm/include/llvm/WindowsResource
     - `3`
     - `1`
     - `2`
     - :part:`33%`
   * - llvm/include/llvm/XRay
     - `17`
     - `14`
     - `3`
     - :part:`82%`
   * - llvm/include/llvm-c
     - `27`
     - `12`
     - `15`
     - :part:`44%`
   * - llvm/include/llvm-c/Transforms
     - `8`
     - `2`
     - `6`
     - :part:`25%`
   * - llvm/lib/Analysis
     - `114`
     - `35`
     - `79`
     - :part:`30%`
   * - llvm/lib/AsmParser
     - `6`
     - `2`
     - `4`
     - :part:`33%`
   * - llvm/lib/BinaryFormat
     - `11`
     - `8`
     - `3`
     - :part:`72%`
   * - llvm/lib/Bitcode/Reader
     - `7`
     - `2`
     - `5`
     - :part:`28%`
   * - llvm/lib/Bitcode/Writer
     - `5`
     - `0`
     - `5`
     - :none:`0%`
   * - llvm/lib/Bitstream/Reader
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/lib/CodeGen
     - `200`
     - `39`
     - `161`
     - :part:`19%`
   * - llvm/lib/CodeGen/AsmPrinter
     - `43`
     - `16`
     - `27`
     - :part:`37%`
   * - llvm/lib/CodeGen/GlobalISel
     - `24`
     - `8`
     - `16`
     - :part:`33%`
   * - llvm/lib/CodeGen/LiveDebugValues
     - `4`
     - `2`
     - `2`
     - :part:`50%`
   * - llvm/lib/CodeGen/MIRParser
     - `4`
     - `1`
     - `3`
     - :part:`25%`
   * - llvm/lib/CodeGen/SelectionDAG
     - `31`
     - `2`
     - `29`
     - :part:`6%`
   * - llvm/lib/DebugInfo/CodeView
     - `40`
     - `24`
     - `16`
     - :part:`60%`
   * - llvm/lib/DebugInfo/DWARF
     - `28`
     - `8`
     - `20`
     - :part:`28%`
   * - llvm/lib/DebugInfo/GSYM
     - `11`
     - `1`
     - `10`
     - :part:`9%`
   * - llvm/lib/DebugInfo/MSF
     - `4`
     - `4`
     - `0`
     - :good:`100%`
   * - llvm/lib/DebugInfo/PDB
     - `40`
     - `34`
     - `6`
     - :part:`85%`
   * - llvm/lib/DebugInfo/PDB/DIA
     - `18`
     - `15`
     - `3`
     - :part:`83%`
   * - llvm/lib/DebugInfo/PDB/Native
     - `50`
     - `37`
     - `13`
     - :part:`74%`
   * - llvm/lib/DebugInfo/Symbolize
     - `4`
     - `1`
     - `3`
     - :part:`25%`
   * - llvm/lib/Demangle
     - `4`
     - `2`
     - `2`
     - :part:`50%`
   * - llvm/lib/DWARFLinker
     - `4`
     - `3`
     - `1`
     - :part:`75%`
   * - llvm/lib/ExecutionEngine
     - `5`
     - `1`
     - `4`
     - :part:`20%`
   * - llvm/lib/ExecutionEngine/IntelJITEvents
     - `5`
     - `0`
     - `5`
     - :none:`0%`
   * - llvm/lib/ExecutionEngine/Interpreter
     - `4`
     - `0`
     - `4`
     - :none:`0%`
   * - llvm/lib/ExecutionEngine/JITLink
     - `14`
     - `9`
     - `5`
     - :part:`64%`
   * - llvm/lib/ExecutionEngine/MCJIT
     - `2`
     - `0`
     - `2`
     - :none:`0%`
   * - llvm/lib/ExecutionEngine/OProfileJIT
     - `2`
     - `0`
     - `2`
     - :none:`0%`
   * - llvm/lib/ExecutionEngine/Orc
     - `26`
     - `15`
     - `11`
     - :part:`57%`
   * - llvm/lib/ExecutionEngine/Orc/Shared
     - `3`
     - `3`
     - `0`
     - :good:`100%`
   * - llvm/lib/ExecutionEngine/Orc/TargetProcess
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - llvm/lib/ExecutionEngine/PerfJITEvents
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/lib/ExecutionEngine/RuntimeDyld
     - `12`
     - `1`
     - `11`
     - :part:`8%`
   * - llvm/lib/ExecutionEngine/RuntimeDyld/Targets
     - `10`
     - `1`
     - `9`
     - :part:`10%`
   * - llvm/lib/Extensions
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/lib/FileCheck
     - `2`
     - `1`
     - `1`
     - :part:`50%`
   * - llvm/lib/Frontend/OpenMP
     - `2`
     - `1`
     - `1`
     - :part:`50%`
   * - llvm/lib/FuzzMutate
     - `5`
     - `2`
     - `3`
     - :part:`40%`
   * - llvm/lib/InterfaceStub
     - `3`
     - `3`
     - `0`
     - :good:`100%`
   * - llvm/lib/IR
     - `63`
     - `13`
     - `50`
     - :part:`20%`
   * - llvm/lib/IRReader
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/lib/LineEditor
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/lib/Linker
     - `3`
     - `0`
     - `3`
     - :none:`0%`
   * - llvm/lib/LTO
     - `8`
     - `1`
     - `7`
     - :part:`12%`
   * - llvm/lib/MC
     - `62`
     - `20`
     - `42`
     - :part:`32%`
   * - llvm/lib/MC/MCDisassembler
     - `6`
     - `3`
     - `3`
     - :part:`50%`
   * - llvm/lib/MC/MCParser
     - `12`
     - `1`
     - `11`
     - :part:`8%`
   * - llvm/lib/MCA
     - `7`
     - `3`
     - `4`
     - :part:`42%`
   * - llvm/lib/MCA/HardwareUnits
     - `6`
     - `3`
     - `3`
     - :part:`50%`
   * - llvm/lib/MCA/Stages
     - `7`
     - `6`
     - `1`
     - :part:`85%`
   * - llvm/lib/Object
     - `30`
     - `12`
     - `18`
     - :part:`40%`
   * - llvm/lib/ObjectYAML
     - `22`
     - `11`
     - `11`
     - :part:`50%`
   * - llvm/lib/Option
     - `4`
     - `0`
     - `4`
     - :none:`0%`
   * - llvm/lib/Passes
     - `3`
     - `2`
     - `1`
     - :part:`66%`
   * - llvm/lib/ProfileData
     - `8`
     - `3`
     - `5`
     - :part:`37%`
   * - llvm/lib/ProfileData/Coverage
     - `3`
     - `0`
     - `3`
     - :none:`0%`
   * - llvm/lib/Remarks
     - `13`
     - `10`
     - `3`
     - :part:`76%`
   * - llvm/lib/Support
     - `132`
     - `46`
     - `86`
     - :part:`34%`
   * - llvm/lib/Support/Unix
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/lib/TableGen
     - `13`
     - `1`
     - `12`
     - :part:`7%`
   * - llvm/lib/Target
     - `5`
     - `0`
     - `5`
     - :none:`0%`
   * - llvm/lib/Target/AArch64
     - `58`
     - `4`
     - `54`
     - :part:`6%`
   * - llvm/lib/Target/AArch64/AsmParser
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/lib/Target/AArch64/Disassembler
     - `4`
     - `1`
     - `3`
     - :part:`25%`
   * - llvm/lib/Target/AArch64/GISel
     - `12`
     - `2`
     - `10`
     - :part:`16%`
   * - llvm/lib/Target/AArch64/MCTargetDesc
     - `21`
     - `6`
     - `15`
     - :part:`28%`
   * - llvm/lib/Target/AArch64/TargetInfo
     - `2`
     - `1`
     - `1`
     - :part:`50%`
   * - llvm/lib/Target/AArch64/Utils
     - `2`
     - `0`
     - `2`
     - :none:`0%`
   * - llvm/lib/Target/AMDGPU
     - `148`
     - `17`
     - `131`
     - :part:`11%`
   * - llvm/lib/Target/AMDGPU/AsmParser
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/lib/Target/AMDGPU/Disassembler
     - `2`
     - `0`
     - `2`
     - :none:`0%`
   * - llvm/lib/Target/AMDGPU/MCTargetDesc
     - `18`
     - `3`
     - `15`
     - :part:`16%`
   * - llvm/lib/Target/AMDGPU/TargetInfo
     - `2`
     - `1`
     - `1`
     - :part:`50%`
   * - llvm/lib/Target/AMDGPU/Utils
     - `9`
     - `2`
     - `7`
     - :part:`22%`
   * - llvm/lib/Target/ARC
     - `24`
     - `19`
     - `5`
     - :part:`79%`
   * - llvm/lib/Target/ARC/Disassembler
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/lib/Target/ARC/MCTargetDesc
     - `7`
     - `6`
     - `1`
     - :part:`85%`
   * - llvm/lib/Target/ARC/TargetInfo
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - llvm/lib/Target/ARM
     - `71`
     - `7`
     - `64`
     - :part:`9%`
   * - llvm/lib/Target/ARM/AsmParser
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/lib/Target/ARM/Disassembler
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/lib/Target/ARM/MCTargetDesc
     - `26`
     - `2`
     - `24`
     - :part:`7%`
   * - llvm/lib/Target/ARM/TargetInfo
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - llvm/lib/Target/ARM/Utils
     - `2`
     - `0`
     - `2`
     - :none:`0%`
   * - llvm/lib/Target/AVR
     - `23`
     - `4`
     - `19`
     - :part:`17%`
   * - llvm/lib/Target/AVR/AsmParser
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/lib/Target/AVR/Disassembler
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/lib/Target/AVR/MCTargetDesc
     - `20`
     - `6`
     - `14`
     - :part:`30%`
   * - llvm/lib/Target/AVR/TargetInfo
     - `2`
     - `1`
     - `1`
     - :part:`50%`
   * - llvm/lib/Target/BPF
     - `30`
     - `7`
     - `23`
     - :part:`23%`
   * - llvm/lib/Target/BPF/AsmParser
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/lib/Target/BPF/Disassembler
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/lib/Target/BPF/MCTargetDesc
     - `8`
     - `1`
     - `7`
     - :part:`12%`
   * - llvm/lib/Target/BPF/TargetInfo
     - `2`
     - `1`
     - `1`
     - :part:`50%`
   * - llvm/lib/Target/CSKY
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - llvm/lib/Target/CSKY/TargetInfo
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - llvm/lib/Target/Hexagon
     - `79`
     - `4`
     - `75`
     - :part:`5%`
   * - llvm/lib/Target/Hexagon/AsmParser
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/lib/Target/Hexagon/Disassembler
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/lib/Target/Hexagon/MCTargetDesc
     - `26`
     - `6`
     - `20`
     - :part:`23%`
   * - llvm/lib/Target/Hexagon/TargetInfo
     - `2`
     - `1`
     - `1`
     - :part:`50%`
   * - llvm/lib/Target/Lanai
     - `28`
     - `19`
     - `9`
     - :part:`67%`
   * - llvm/lib/Target/Lanai/AsmParser
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/lib/Target/Lanai/Disassembler
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - llvm/lib/Target/Lanai/MCTargetDesc
     - `13`
     - `12`
     - `1`
     - :part:`92%`
   * - llvm/lib/Target/Lanai/TargetInfo
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - llvm/lib/Target/Mips
     - `69`
     - `12`
     - `57`
     - :part:`17%`
   * - llvm/lib/Target/Mips/AsmParser
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/lib/Target/Mips/Disassembler
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/lib/Target/Mips/MCTargetDesc
     - `25`
     - `6`
     - `19`
     - :part:`24%`
   * - llvm/lib/Target/Mips/TargetInfo
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - llvm/lib/Target/MSP430
     - `20`
     - `0`
     - `20`
     - :none:`0%`
   * - llvm/lib/Target/MSP430/AsmParser
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/lib/Target/MSP430/Disassembler
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/lib/Target/MSP430/MCTargetDesc
     - `11`
     - `3`
     - `8`
     - :part:`27%`
   * - llvm/lib/Target/MSP430/TargetInfo
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - llvm/lib/Target/NVPTX
     - `42`
     - `7`
     - `35`
     - :part:`16%`
   * - llvm/lib/Target/NVPTX/MCTargetDesc
     - `9`
     - `5`
     - `4`
     - :part:`55%`
   * - llvm/lib/Target/NVPTX/TargetInfo
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - llvm/lib/Target/PowerPC
     - `52`
     - `2`
     - `50`
     - :part:`3%`
   * - llvm/lib/Target/PowerPC/AsmParser
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/lib/Target/PowerPC/Disassembler
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/lib/Target/PowerPC/GISel
     - `7`
     - `7`
     - `0`
     - :good:`100%`
   * - llvm/lib/Target/PowerPC/MCTargetDesc
     - `18`
     - `3`
     - `15`
     - :part:`16%`
   * - llvm/lib/Target/PowerPC/TargetInfo
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - llvm/lib/Target/RISCV
     - `32`
     - `15`
     - `17`
     - :part:`46%`
   * - llvm/lib/Target/RISCV/AsmParser
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/lib/Target/RISCV/Disassembler
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/lib/Target/RISCV/MCTargetDesc
     - `17`
     - `8`
     - `9`
     - :part:`47%`
   * - llvm/lib/Target/RISCV/TargetInfo
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - llvm/lib/Target/RISCV/Utils
     - `4`
     - `3`
     - `1`
     - :part:`75%`
   * - llvm/lib/Target/Sparc
     - `23`
     - `2`
     - `21`
     - :part:`8%`
   * - llvm/lib/Target/Sparc/AsmParser
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/lib/Target/Sparc/Disassembler
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/lib/Target/Sparc/MCTargetDesc
     - `14`
     - `4`
     - `10`
     - :part:`28%`
   * - llvm/lib/Target/Sparc/TargetInfo
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - llvm/lib/Target/SystemZ
     - `40`
     - `4`
     - `36`
     - :part:`10%`
   * - llvm/lib/Target/SystemZ/AsmParser
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/lib/Target/SystemZ/Disassembler
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/lib/Target/SystemZ/MCTargetDesc
     - `10`
     - `4`
     - `6`
     - :part:`40%`
   * - llvm/lib/Target/SystemZ/TargetInfo
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - llvm/lib/Target/VE
     - `20`
     - `16`
     - `4`
     - :part:`80%`
   * - llvm/lib/Target/VE/AsmParser
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - llvm/lib/Target/VE/Disassembler
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - llvm/lib/Target/VE/MCTargetDesc
     - `14`
     - `13`
     - `1`
     - :part:`92%`
   * - llvm/lib/Target/VE/TargetInfo
     - `2`
     - `1`
     - `1`
     - :part:`50%`
   * - llvm/lib/Target/WebAssembly
     - `60`
     - `43`
     - `17`
     - :part:`71%`
   * - llvm/lib/Target/WebAssembly/AsmParser
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/lib/Target/WebAssembly/Disassembler
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - llvm/lib/Target/WebAssembly/MCTargetDesc
     - `12`
     - `7`
     - `5`
     - :part:`58%`
   * - llvm/lib/Target/WebAssembly/TargetInfo
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - llvm/lib/Target/X86
     - `75`
     - `12`
     - `63`
     - :part:`16%`
   * - llvm/lib/Target/X86/AsmParser
     - `3`
     - `0`
     - `3`
     - :none:`0%`
   * - llvm/lib/Target/X86/Disassembler
     - `2`
     - `0`
     - `2`
     - :none:`0%`
   * - llvm/lib/Target/X86/MCTargetDesc
     - `25`
     - `5`
     - `20`
     - :part:`20%`
   * - llvm/lib/Target/X86/TargetInfo
     - `2`
     - `1`
     - `1`
     - :part:`50%`
   * - llvm/lib/Target/XCore
     - `27`
     - `2`
     - `25`
     - :part:`7%`
   * - llvm/lib/Target/XCore/Disassembler
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/lib/Target/XCore/MCTargetDesc
     - `6`
     - `3`
     - `3`
     - :part:`50%`
   * - llvm/lib/Target/XCore/TargetInfo
     - `2`
     - `1`
     - `1`
     - :part:`50%`
   * - llvm/lib/Testing/Support
     - `3`
     - `3`
     - `0`
     - :good:`100%`
   * - llvm/lib/TextAPI
     - `11`
     - `8`
     - `3`
     - :part:`72%`
   * - llvm/lib/ToolDrivers/llvm-dlltool
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/lib/ToolDrivers/llvm-lib
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/lib/Transforms/AggressiveInstCombine
     - `3`
     - `0`
     - `3`
     - :none:`0%`
   * - llvm/lib/Transforms/CFGuard
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - llvm/lib/Transforms/Coroutines
     - `8`
     - `0`
     - `8`
     - :none:`0%`
   * - llvm/lib/Transforms/Hello
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/lib/Transforms/HelloNew
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - llvm/lib/Transforms/InstCombine
     - `16`
     - `1`
     - `15`
     - :part:`6%`
   * - llvm/lib/Transforms/Instrumentation
     - `22`
     - `5`
     - `17`
     - :part:`22%`
   * - llvm/lib/Transforms/IPO
     - `40`
     - `7`
     - `33`
     - :part:`17%`
   * - llvm/lib/Transforms/ObjCARC
     - `15`
     - `4`
     - `11`
     - :part:`26%`
   * - llvm/lib/Transforms/Scalar
     - `77`
     - `12`
     - `65`
     - :part:`15%`
   * - llvm/lib/Transforms/Utils
     - `72`
     - `14`
     - `58`
     - :part:`19%`
   * - llvm/lib/Transforms/Vectorize
     - `22`
     - `14`
     - `8`
     - :part:`63%`
   * - llvm/lib/WindowsManifest
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - llvm/lib/XRay
     - `14`
     - `12`
     - `2`
     - :part:`85%`
   * - llvm/tools/bugpoint
     - `12`
     - `1`
     - `11`
     - :part:`8%`
   * - llvm/tools/bugpoint-passes
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/tools/dsymutil
     - `18`
     - `15`
     - `3`
     - :part:`83%`
   * - llvm/tools/gold
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/tools/llc
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/tools/lli
     - `2`
     - `0`
     - `2`
     - :none:`0%`
   * - llvm/tools/lli/ChildTarget
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - llvm/tools/llvm-ar
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/tools/llvm-as
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/tools/llvm-as-fuzzer
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/tools/llvm-bcanalyzer
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - llvm/tools/llvm-c-test
     - `2`
     - `0`
     - `2`
     - :none:`0%`
   * - llvm/tools/llvm-cat
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/tools/llvm-cfi-verify
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/tools/llvm-cfi-verify/lib
     - `4`
     - `1`
     - `3`
     - :part:`25%`
   * - llvm/tools/llvm-config
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/tools/llvm-cov
     - `23`
     - `12`
     - `11`
     - :part:`52%`
   * - llvm/tools/llvm-cvtres
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/tools/llvm-cxxdump
     - `4`
     - `2`
     - `2`
     - :part:`50%`
   * - llvm/tools/llvm-cxxfilt
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/tools/llvm-cxxmap
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/tools/llvm-diff
     - `7`
     - `0`
     - `7`
     - :none:`0%`
   * - llvm/tools/llvm-dis
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/tools/llvm-dwarfdump
     - `4`
     - `2`
     - `2`
     - :part:`50%`
   * - llvm/tools/llvm-dwarfdump/fuzzer
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/tools/llvm-dwp
     - `4`
     - `1`
     - `3`
     - :part:`25%`
   * - llvm/tools/llvm-elfabi
     - `3`
     - `3`
     - `0`
     - :good:`100%`
   * - llvm/tools/llvm-exegesis
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - llvm/tools/llvm-exegesis/lib
     - `44`
     - `35`
     - `9`
     - :part:`79%`
   * - llvm/tools/llvm-exegesis/lib/AArch64
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - llvm/tools/llvm-exegesis/lib/Mips
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/tools/llvm-exegesis/lib/PowerPC
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - llvm/tools/llvm-exegesis/lib/X86
     - `3`
     - `2`
     - `1`
     - :part:`66%`
   * - llvm/tools/llvm-extract
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/tools/llvm-gsymutil
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/tools/llvm-ifs
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - llvm/tools/llvm-isel-fuzzer
     - `2`
     - `1`
     - `1`
     - :part:`50%`
   * - llvm/tools/llvm-itanium-demangle-fuzzer
     - `2`
     - `1`
     - `1`
     - :part:`50%`
   * - llvm/tools/llvm-jitlink
     - `4`
     - `2`
     - `2`
     - :part:`50%`
   * - llvm/tools/llvm-jitlink/llvm-jitlink-executor
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - llvm/tools/llvm-jitlistener
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/tools/llvm-libtool-darwin
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - llvm/tools/llvm-link
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/tools/llvm-lipo
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/tools/llvm-lto
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/tools/llvm-lto2
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/tools/llvm-mc
     - `3`
     - `1`
     - `2`
     - :part:`33%`
   * - llvm/tools/llvm-mc-assemble-fuzzer
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/tools/llvm-mc-disassemble-fuzzer
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/tools/llvm-mca
     - `7`
     - `6`
     - `1`
     - :part:`85%`
   * - llvm/tools/llvm-mca/Views
     - `20`
     - `13`
     - `7`
     - :part:`65%`
   * - llvm/tools/llvm-microsoft-demangle-fuzzer
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - llvm/tools/llvm-ml
     - `3`
     - `1`
     - `2`
     - :part:`33%`
   * - llvm/tools/llvm-modextract
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - llvm/tools/llvm-mt
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/tools/llvm-nm
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/tools/llvm-objcopy
     - `6`
     - `4`
     - `2`
     - :part:`66%`
   * - llvm/tools/llvm-objcopy/COFF
     - `8`
     - `8`
     - `0`
     - :good:`100%`
   * - llvm/tools/llvm-objcopy/ELF
     - `6`
     - `5`
     - `1`
     - :part:`83%`
   * - llvm/tools/llvm-objcopy/MachO
     - `10`
     - `10`
     - `0`
     - :good:`100%`
   * - llvm/tools/llvm-objcopy/wasm
     - `8`
     - `8`
     - `0`
     - :good:`100%`
   * - llvm/tools/llvm-objdump
     - `12`
     - `8`
     - `4`
     - :part:`66%`
   * - llvm/tools/llvm-opt-fuzzer
     - `2`
     - `0`
     - `2`
     - :none:`0%`
   * - llvm/tools/llvm-opt-report
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/tools/llvm-pdbutil
     - `47`
     - `16`
     - `31`
     - :part:`34%`
   * - llvm/tools/llvm-profdata
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/tools/llvm-profgen
     - `7`
     - `7`
     - `0`
     - :good:`100%`
   * - llvm/tools/llvm-rc
     - `12`
     - `7`
     - `5`
     - :part:`58%`
   * - llvm/tools/llvm-readobj
     - `19`
     - `4`
     - `15`
     - :part:`21%`
   * - llvm/tools/llvm-reduce
     - `4`
     - `3`
     - `1`
     - :part:`75%`
   * - llvm/tools/llvm-reduce/deltas
     - `24`
     - `21`
     - `3`
     - :part:`87%`
   * - llvm/tools/llvm-rtdyld
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/tools/llvm-shlib
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - llvm/tools/llvm-size
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/tools/llvm-special-case-list-fuzzer
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - llvm/tools/llvm-split
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/tools/llvm-stress
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/tools/llvm-strings
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/tools/llvm-symbolizer
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/tools/llvm-undname
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - llvm/tools/llvm-xray
     - `19`
     - `16`
     - `3`
     - :part:`84%`
   * - llvm/tools/llvm-yaml-numeric-parser-fuzzer
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - llvm/tools/llvm-yaml-parser-fuzzer
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - llvm/tools/lto
     - `2`
     - `1`
     - `1`
     - :part:`50%`
   * - llvm/tools/obj2yaml
     - `10`
     - `5`
     - `5`
     - :part:`50%`
   * - llvm/tools/opt
     - `10`
     - `3`
     - `7`
     - :part:`30%`
   * - llvm/tools/remarks-shlib
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/tools/sancov
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/tools/sanstats
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - llvm/tools/split-file
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - llvm/tools/verify-uselistorder
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/tools/vfabi-demangle-fuzzer
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - llvm/tools/yaml2obj
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - llvm/unittests/ADT
     - `76`
     - `31`
     - `45`
     - :part:`40%`
   * - llvm/unittests/Analysis
     - `36`
     - `11`
     - `25`
     - :part:`30%`
   * - llvm/unittests/AsmParser
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - llvm/unittests/BinaryFormat
     - `6`
     - `5`
     - `1`
     - :part:`83%`
   * - llvm/unittests/Bitcode
     - `2`
     - `1`
     - `1`
     - :part:`50%`
   * - llvm/unittests/Bitstream
     - `2`
     - `1`
     - `1`
     - :part:`50%`
   * - llvm/unittests/CodeGen
     - `17`
     - `8`
     - `9`
     - :part:`47%`
   * - llvm/unittests/CodeGen/GlobalISel
     - `11`
     - `1`
     - `10`
     - :part:`9%`
   * - llvm/unittests/DebugInfo/CodeView
     - `3`
     - `1`
     - `2`
     - :part:`33%`
   * - llvm/unittests/DebugInfo/DWARF
     - `15`
     - `9`
     - `6`
     - :part:`60%`
   * - llvm/unittests/DebugInfo/GSYM
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/unittests/DebugInfo/MSF
     - `3`
     - `2`
     - `1`
     - :part:`66%`
   * - llvm/unittests/DebugInfo/PDB
     - `5`
     - `3`
     - `2`
     - :part:`60%`
   * - llvm/unittests/DebugInfo/PDB/Inputs
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - llvm/unittests/Demangle
     - `3`
     - `2`
     - `1`
     - :part:`66%`
   * - llvm/unittests/ExecutionEngine
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/unittests/ExecutionEngine/JITLink
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - llvm/unittests/ExecutionEngine/MCJIT
     - `7`
     - `0`
     - `7`
     - :none:`0%`
   * - llvm/unittests/ExecutionEngine/Orc
     - `13`
     - `4`
     - `9`
     - :part:`30%`
   * - llvm/unittests/FileCheck
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/unittests/Frontend
     - `4`
     - `3`
     - `1`
     - :part:`75%`
   * - llvm/unittests/FuzzMutate
     - `4`
     - `0`
     - `4`
     - :none:`0%`
   * - llvm/unittests/InterfaceStub
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - llvm/unittests/IR
     - `36`
     - `6`
     - `30`
     - :part:`16%`
   * - llvm/unittests/LineEditor
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/unittests/Linker
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/unittests/MC
     - `6`
     - `3`
     - `3`
     - :part:`50%`
   * - llvm/unittests/MC/AMDGPU
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - llvm/unittests/MI
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/unittests/Object
     - `9`
     - `6`
     - `3`
     - :part:`66%`
   * - llvm/unittests/ObjectYAML
     - `5`
     - `3`
     - `2`
     - :part:`60%`
   * - llvm/unittests/Option
     - `2`
     - `1`
     - `1`
     - :part:`50%`
   * - llvm/unittests/Passes
     - `3`
     - `3`
     - `0`
     - :good:`100%`
   * - llvm/unittests/ProfileData
     - `4`
     - `1`
     - `3`
     - :part:`25%`
   * - llvm/unittests/Remarks
     - `8`
     - `5`
     - `3`
     - :part:`62%`
   * - llvm/unittests/Support
     - `93`
     - `28`
     - `65`
     - :part:`30%`
   * - llvm/unittests/Support/DynamicLibrary
     - `4`
     - `0`
     - `4`
     - :none:`0%`
   * - llvm/unittests/TableGen
     - `2`
     - `0`
     - `2`
     - :none:`0%`
   * - llvm/unittests/Target/AArch64
     - `2`
     - `0`
     - `2`
     - :none:`0%`
   * - llvm/unittests/Target/AMDGPU
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - llvm/unittests/Target/ARM
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/unittests/Target/PowerPC
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - llvm/unittests/Target/WebAssembly
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/unittests/Target/X86
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/unittests/TextAPI
     - `5`
     - `3`
     - `2`
     - :part:`60%`
   * - llvm/unittests/tools/llvm-cfi-verify
     - `2`
     - `1`
     - `1`
     - :part:`50%`
   * - llvm/unittests/tools/llvm-exegesis
     - `5`
     - `4`
     - `1`
     - :part:`80%`
   * - llvm/unittests/tools/llvm-exegesis/AArch64
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - llvm/unittests/tools/llvm-exegesis/ARM
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - llvm/unittests/tools/llvm-exegesis/Common
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - llvm/unittests/tools/llvm-exegesis/Mips
     - `5`
     - `4`
     - `1`
     - :part:`80%`
   * - llvm/unittests/tools/llvm-exegesis/PowerPC
     - `4`
     - `2`
     - `2`
     - :part:`50%`
   * - llvm/unittests/tools/llvm-exegesis/X86
     - `9`
     - `8`
     - `1`
     - :part:`88%`
   * - llvm/unittests/Transforms/IPO
     - `4`
     - `2`
     - `2`
     - :part:`50%`
   * - llvm/unittests/Transforms/Scalar
     - `2`
     - `0`
     - `2`
     - :none:`0%`
   * - llvm/unittests/Transforms/Utils
     - `17`
     - `7`
     - `10`
     - :part:`41%`
   * - llvm/unittests/Transforms/Vectorize
     - `7`
     - `7`
     - `0`
     - :good:`100%`
   * - llvm/unittests/XRay
     - `8`
     - `7`
     - `1`
     - :part:`87%`
   * - llvm/utils/benchmark/cmake
     - `5`
     - `3`
     - `2`
     - :part:`60%`
   * - llvm/utils/benchmark/include/benchmark
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/utils/benchmark/src
     - `19`
     - `0`
     - `19`
     - :none:`0%`
   * - llvm/utils/FileCheck
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/utils/fpcmp
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/utils/KillTheDoctor
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/utils/not
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - llvm/utils/PerfectShuffle
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/utils/TableGen
     - `75`
     - `8`
     - `67`
     - :part:`10%`
   * - llvm/utils/TableGen/GlobalISel
     - `17`
     - `8`
     - `9`
     - :part:`47%`
   * - llvm/utils/unittest/googlemock/include/gmock
     - `11`
     - `0`
     - `11`
     - :none:`0%`
   * - llvm/utils/unittest/googlemock/include/gmock/internal
     - `3`
     - `0`
     - `3`
     - :none:`0%`
   * - llvm/utils/unittest/googlemock/include/gmock/internal/custom
     - `3`
     - `0`
     - `3`
     - :none:`0%`
   * - llvm/utils/unittest/googletest/include/gtest
     - `10`
     - `0`
     - `10`
     - :none:`0%`
   * - llvm/utils/unittest/googletest/include/gtest/internal
     - `11`
     - `0`
     - `11`
     - :none:`0%`
   * - llvm/utils/unittest/googletest/include/gtest/internal/custom
     - `4`
     - `0`
     - `4`
     - :none:`0%`
   * - llvm/utils/unittest/googletest/src
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/utils/unittest/UnitTestMain
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/utils/yaml-bench
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - mlir/examples/standalone/include/Standalone
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - mlir/examples/standalone/lib/Standalone
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - mlir/examples/standalone/standalone-opt
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/examples/standalone/standalone-translate
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/examples/toy/Ch1
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/examples/toy/Ch1/include/toy
     - `3`
     - `3`
     - `0`
     - :good:`100%`
   * - mlir/examples/toy/Ch1/parser
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - mlir/examples/toy/Ch2
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/examples/toy/Ch2/include/toy
     - `5`
     - `5`
     - `0`
     - :good:`100%`
   * - mlir/examples/toy/Ch2/mlir
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - mlir/examples/toy/Ch2/parser
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - mlir/examples/toy/Ch3
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/examples/toy/Ch3/include/toy
     - `5`
     - `5`
     - `0`
     - :good:`100%`
   * - mlir/examples/toy/Ch3/mlir
     - `3`
     - `3`
     - `0`
     - :good:`100%`
   * - mlir/examples/toy/Ch3/parser
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - mlir/examples/toy/Ch4
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/examples/toy/Ch4/include/toy
     - `7`
     - `7`
     - `0`
     - :good:`100%`
   * - mlir/examples/toy/Ch4/mlir
     - `4`
     - `4`
     - `0`
     - :good:`100%`
   * - mlir/examples/toy/Ch4/parser
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - mlir/examples/toy/Ch5
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/examples/toy/Ch5/include/toy
     - `7`
     - `7`
     - `0`
     - :good:`100%`
   * - mlir/examples/toy/Ch5/mlir
     - `5`
     - `4`
     - `1`
     - :part:`80%`
   * - mlir/examples/toy/Ch5/parser
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - mlir/examples/toy/Ch6
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/examples/toy/Ch6/include/toy
     - `7`
     - `7`
     - `0`
     - :good:`100%`
   * - mlir/examples/toy/Ch6/mlir
     - `6`
     - `5`
     - `1`
     - :part:`83%`
   * - mlir/examples/toy/Ch6/parser
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - mlir/examples/toy/Ch7
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/examples/toy/Ch7/include/toy
     - `7`
     - `7`
     - `0`
     - :good:`100%`
   * - mlir/examples/toy/Ch7/mlir
     - `6`
     - `5`
     - `1`
     - :part:`83%`
   * - mlir/examples/toy/Ch7/parser
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - mlir/include/mlir
     - `5`
     - `5`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Analysis
     - `11`
     - `10`
     - `1`
     - :part:`90%`
   * - mlir/include/mlir/Analysis/Presburger
     - `3`
     - `3`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/CAPI
     - `8`
     - `8`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Conversion
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Conversion/AffineToStandard
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Conversion/AsyncToLLVM
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Conversion/AVX512ToLLVM
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Conversion/GPUCommon
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Conversion/GPUToNVVM
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - mlir/include/mlir/Conversion/GPUToROCDL
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Conversion/GPUToSPIRV
     - `2`
     - `1`
     - `1`
     - :part:`50%`
   * - mlir/include/mlir/Conversion/GPUToVulkan
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - mlir/include/mlir/Conversion/LinalgToLLVM
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - mlir/include/mlir/Conversion/LinalgToSPIRV
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Conversion/LinalgToStandard
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Conversion/OpenMPToLLVM
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - mlir/include/mlir/Conversion/PDLToPDLInterp
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Conversion/SCFToGPU
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Conversion/SCFToOpenMP
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Conversion/SCFToSPIRV
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Conversion/SCFToStandard
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Conversion/ShapeToStandard
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Conversion/SPIRVToLLVM
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Conversion/StandardToLLVM
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Conversion/StandardToSPIRV
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Conversion/VectorToLLVM
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Conversion/VectorToROCDL
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Conversion/VectorToSCF
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Conversion/VectorToSPIRV
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Dialect
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Dialect/Affine
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Dialect/Affine/EDSC
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Dialect/Affine/IR
     - `3`
     - `3`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Dialect/Async
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Dialect/Async/IR
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Dialect/AVX512
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Dialect/GPU
     - `5`
     - `5`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Dialect/Linalg
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Dialect/Linalg/Analysis
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Dialect/Linalg/EDSC
     - `3`
     - `3`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Dialect/Linalg/IR
     - `3`
     - `2`
     - `1`
     - :part:`66%`
   * - mlir/include/mlir/Dialect/Linalg/Transforms
     - `3`
     - `3`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Dialect/Linalg/Utils
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Dialect/LLVMIR
     - `5`
     - `5`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Dialect/LLVMIR/Transforms
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Dialect/OpenACC
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Dialect/OpenMP
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Dialect/PDL/IR
     - `2`
     - `1`
     - `1`
     - :part:`50%`
   * - mlir/include/mlir/Dialect/PDLInterp/IR
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Dialect/Quant
     - `6`
     - `6`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Dialect/SCF
     - `4`
     - `3`
     - `1`
     - :part:`75%`
   * - mlir/include/mlir/Dialect/SCF/EDSC
     - `2`
     - `1`
     - `1`
     - :part:`50%`
   * - mlir/include/mlir/Dialect/SDBM
     - `3`
     - `2`
     - `1`
     - :part:`66%`
   * - mlir/include/mlir/Dialect/Shape/IR
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Dialect/Shape/Transforms
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Dialect/SPIRV
     - `13`
     - `12`
     - `1`
     - :part:`92%`
   * - mlir/include/mlir/Dialect/StandardOps/EDSC
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Dialect/StandardOps/IR
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Dialect/StandardOps/Transforms
     - `3`
     - `3`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Dialect/Tosa/IR
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Dialect/Tosa/Transforms
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Dialect/Tosa/Utils
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Dialect/Utils
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Dialect/Vector
     - `3`
     - `3`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Dialect/Vector/EDSC
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/EDSC
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/ExecutionEngine
     - `6`
     - `4`
     - `2`
     - :part:`66%`
   * - mlir/include/mlir/Interfaces
     - `11`
     - `10`
     - `1`
     - :part:`90%`
   * - mlir/include/mlir/IR
     - `46`
     - `15`
     - `31`
     - :part:`32%`
   * - mlir/include/mlir/Pass
     - `6`
     - `0`
     - `6`
     - :none:`0%`
   * - mlir/include/mlir/Reducer
     - `6`
     - `6`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Reducer/Passes
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Rewrite
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Support
     - `12`
     - `6`
     - `6`
     - :part:`50%`
   * - mlir/include/mlir/TableGen
     - `20`
     - `18`
     - `2`
     - :part:`90%`
   * - mlir/include/mlir/Target
     - `3`
     - `3`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Target/LLVMIR
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Transforms
     - `14`
     - `9`
     - `5`
     - :part:`64%`
   * - mlir/include/mlir-c
     - `11`
     - `11`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir-c/Bindings/Python
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/lib/Analysis
     - `11`
     - `10`
     - `1`
     - :part:`90%`
   * - mlir/lib/Analysis/Presburger
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - mlir/lib/Bindings/Python
     - `8`
     - `8`
     - `0`
     - :good:`100%`
   * - mlir/lib/Bindings/Python/Transforms
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/lib/CAPI/IR
     - `8`
     - `8`
     - `0`
     - :good:`100%`
   * - mlir/lib/CAPI/Registration
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/lib/CAPI/Standard
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/lib/CAPI/Transforms
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/lib/Conversion
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/lib/Conversion/AffineToStandard
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - mlir/lib/Conversion/AsyncToLLVM
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/lib/Conversion/AVX512ToLLVM
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/lib/Conversion/GPUCommon
     - `5`
     - `5`
     - `0`
     - :good:`100%`
   * - mlir/lib/Conversion/GPUToNVVM
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/lib/Conversion/GPUToROCDL
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/lib/Conversion/GPUToSPIRV
     - `2`
     - `1`
     - `1`
     - :part:`50%`
   * - mlir/lib/Conversion/GPUToVulkan
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - mlir/lib/Conversion/LinalgToLLVM
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/lib/Conversion/LinalgToSPIRV
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - mlir/lib/Conversion/LinalgToStandard
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/lib/Conversion/OpenMPToLLVM
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/lib/Conversion/PDLToPDLInterp
     - `5`
     - `4`
     - `1`
     - :part:`80%`
   * - mlir/lib/Conversion/SCFToGPU
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - mlir/lib/Conversion/SCFToOpenMP
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/lib/Conversion/SCFToSPIRV
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/lib/Conversion/SCFToStandard
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/lib/Conversion/ShapeToStandard
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - mlir/lib/Conversion/SPIRVToLLVM
     - `3`
     - `2`
     - `1`
     - :part:`66%`
   * - mlir/lib/Conversion/StandardToLLVM
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - mlir/lib/Conversion/StandardToSPIRV
     - `3`
     - `3`
     - `0`
     - :good:`100%`
   * - mlir/lib/Conversion/VectorToLLVM
     - `2`
     - `1`
     - `1`
     - :part:`50%`
   * - mlir/lib/Conversion/VectorToROCDL
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/lib/Conversion/VectorToSCF
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/lib/Conversion/VectorToSPIRV
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - mlir/lib/Dialect
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/lib/Dialect/Affine/EDSC
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/lib/Dialect/Affine/IR
     - `3`
     - `3`
     - `0`
     - :good:`100%`
   * - mlir/lib/Dialect/Affine/Transforms
     - `10`
     - `10`
     - `0`
     - :good:`100%`
   * - mlir/lib/Dialect/Affine/Utils
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/lib/Dialect/Async/IR
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/lib/Dialect/Async/Transforms
     - `4`
     - `4`
     - `0`
     - :good:`100%`
   * - mlir/lib/Dialect/AVX512/IR
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/lib/Dialect/GPU/IR
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - mlir/lib/Dialect/GPU/Transforms
     - `6`
     - `5`
     - `1`
     - :part:`83%`
   * - mlir/lib/Dialect/Linalg/Analysis
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/lib/Dialect/Linalg/EDSC
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/lib/Dialect/Linalg/IR
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - mlir/lib/Dialect/Linalg/Transforms
     - `16`
     - `16`
     - `0`
     - :good:`100%`
   * - mlir/lib/Dialect/Linalg/Utils
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/lib/Dialect/LLVMIR/IR
     - `7`
     - `6`
     - `1`
     - :part:`85%`
   * - mlir/lib/Dialect/LLVMIR/Transforms
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - mlir/lib/Dialect/OpenACC/IR
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/lib/Dialect/OpenMP/IR
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/lib/Dialect/PDL/IR
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/lib/Dialect/PDLInterp/IR
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/lib/Dialect/Quant/IR
     - `4`
     - `4`
     - `0`
     - :good:`100%`
   * - mlir/lib/Dialect/Quant/Transforms
     - `3`
     - `3`
     - `0`
     - :good:`100%`
   * - mlir/lib/Dialect/Quant/Utils
     - `3`
     - `3`
     - `0`
     - :good:`100%`
   * - mlir/lib/Dialect/SCF
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/lib/Dialect/SCF/EDSC
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - mlir/lib/Dialect/SCF/Transforms
     - `7`
     - `7`
     - `0`
     - :good:`100%`
   * - mlir/lib/Dialect/SDBM
     - `4`
     - `4`
     - `0`
     - :good:`100%`
   * - mlir/lib/Dialect/Shape/IR
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/lib/Dialect/Shape/Transforms
     - `5`
     - `5`
     - `0`
     - :good:`100%`
   * - mlir/lib/Dialect/SPIRV
     - `8`
     - `6`
     - `2`
     - :part:`75%`
   * - mlir/lib/Dialect/SPIRV/Linking/ModuleCombiner
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/lib/Dialect/SPIRV/Serialization
     - `4`
     - `4`
     - `0`
     - :good:`100%`
   * - mlir/lib/Dialect/SPIRV/Transforms
     - `5`
     - `5`
     - `0`
     - :good:`100%`
   * - mlir/lib/Dialect/StandardOps/EDSC
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - mlir/lib/Dialect/StandardOps/IR
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/lib/Dialect/StandardOps/Transforms
     - `8`
     - `8`
     - `0`
     - :good:`100%`
   * - mlir/lib/Dialect/Tosa/IR
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/lib/Dialect/Tosa/Transforms
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/lib/Dialect/Tosa/Utils
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/lib/Dialect/Vector
     - `4`
     - `3`
     - `1`
     - :part:`75%`
   * - mlir/lib/Dialect/Vector/EDSC
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/lib/EDSC
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - mlir/lib/ExecutionEngine
     - `7`
     - `7`
     - `0`
     - :good:`100%`
   * - mlir/lib/Interfaces
     - `9`
     - `9`
     - `0`
     - :good:`100%`
   * - mlir/lib/IR
     - `35`
     - `34`
     - `1`
     - :part:`97%`
   * - mlir/lib/Parser
     - `12`
     - `12`
     - `0`
     - :good:`100%`
   * - mlir/lib/Pass
     - `7`
     - `6`
     - `1`
     - :part:`85%`
   * - mlir/lib/Reducer
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/lib/Rewrite
     - `4`
     - `3`
     - `1`
     - :part:`75%`
   * - mlir/lib/Support
     - `5`
     - `5`
     - `0`
     - :good:`100%`
   * - mlir/lib/TableGen
     - `17`
     - `15`
     - `2`
     - :part:`88%`
   * - mlir/lib/Target/LLVMIR
     - `9`
     - `9`
     - `0`
     - :good:`100%`
   * - mlir/lib/Transforms
     - `24`
     - `19`
     - `5`
     - :part:`79%`
   * - mlir/lib/Transforms/Utils
     - `8`
     - `6`
     - `2`
     - :part:`75%`
   * - mlir/lib/Translation
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/tools/mlir-cpu-runner
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/tools/mlir-cuda-runner
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - mlir/tools/mlir-linalg-ods-gen
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/tools/mlir-opt
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/tools/mlir-reduce
     - `4`
     - `3`
     - `1`
     - :part:`75%`
   * - mlir/tools/mlir-reduce/Passes
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/tools/mlir-rocm-runner
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - mlir/tools/mlir-shlib
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/tools/mlir-spirv-cpu-runner
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/tools/mlir-tblgen
     - `20`
     - `19`
     - `1`
     - :part:`95%`
   * - mlir/tools/mlir-translate
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/tools/mlir-vulkan-runner
     - `4`
     - `3`
     - `1`
     - :part:`75%`
   * - mlir/unittests/Analysis
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - mlir/unittests/Analysis/Presburger
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - mlir/unittests/Dialect
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/unittests/Dialect/Quant
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/unittests/Dialect/SPIRV
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - mlir/unittests/IR
     - `3`
     - `3`
     - `0`
     - :good:`100%`
   * - mlir/unittests/Pass
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - mlir/unittests/SDBM
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - mlir/unittests/Support
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - mlir/unittests/TableGen
     - `4`
     - `2`
     - `2`
     - :part:`50%`
   * - openmp/libomptarget/deviceRTLs
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - openmp/libomptarget/deviceRTLs/amdgcn/src
     - `3`
     - `3`
     - `0`
     - :good:`100%`
   * - openmp/libomptarget/deviceRTLs/common
     - `8`
     - `4`
     - `4`
     - :part:`50%`
   * - openmp/libomptarget/deviceRTLs/nvptx/src
     - `2`
     - `1`
     - `1`
     - :part:`50%`
   * - openmp/libomptarget/include
     - `4`
     - `2`
     - `2`
     - :part:`50%`
   * - openmp/libomptarget/plugins/amdgpu/impl
     - `14`
     - `14`
     - `0`
     - :good:`100%`
   * - openmp/libomptarget/plugins/amdgpu/src
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - openmp/libomptarget/plugins/cuda/src
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - openmp/libomptarget/plugins/generic-elf-64bit/src
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - openmp/libomptarget/plugins/ve/src
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - openmp/libomptarget/src
     - `10`
     - `2`
     - `8`
     - :part:`20%`
   * - openmp/runtime/doc/doxygen
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - openmp/runtime/src
     - `74`
     - `36`
     - `38`
     - :part:`48%`
   * - openmp/runtime/src/thirdparty/ittnotify
     - `6`
     - `0`
     - `6`
     - :none:`0%`
   * - openmp/runtime/src/thirdparty/ittnotify/legacy
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - openmp/tools/archer
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - openmp/tools/archer/tests/ompt
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - openmp/tools/multiplex
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - openmp/tools/multiplex/tests
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - openmp/tools/multiplex/tests/custom_data_storage
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - openmp/tools/multiplex/tests/print
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - parallel-libs/acxxel
     - `6`
     - `4`
     - `2`
     - :part:`66%`
   * - parallel-libs/acxxel/examples
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - parallel-libs/acxxel/tests
     - `5`
     - `4`
     - `1`
     - :part:`80%`
   * - polly/include/polly
     - `22`
     - `21`
     - `1`
     - :part:`95%`
   * - polly/include/polly/CodeGen
     - `14`
     - `14`
     - `0`
     - :good:`100%`
   * - polly/include/polly/Support
     - `11`
     - `11`
     - `0`
     - :good:`100%`
   * - polly/lib/Analysis
     - `9`
     - `8`
     - `1`
     - :part:`88%`
   * - polly/lib/CodeGen
     - `15`
     - `15`
     - `0`
     - :good:`100%`
   * - polly/lib/Exchange
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - polly/lib/External/isl
     - `67`
     - `1`
     - `66`
     - :part:`1%`
   * - polly/lib/External/isl/imath
     - `3`
     - `0`
     - `3`
     - :none:`0%`
   * - polly/lib/External/isl/imath_wrap
     - `4`
     - `0`
     - `4`
     - :none:`0%`
   * - polly/lib/External/isl/include/isl
     - `62`
     - `8`
     - `54`
     - :part:`12%`
   * - polly/lib/External/isl/interface
     - `5`
     - `1`
     - `4`
     - :part:`20%`
   * - polly/lib/External/pet/include
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - polly/lib/External/ppcg
     - `17`
     - `0`
     - `17`
     - :none:`0%`
   * - polly/lib/Plugin
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - polly/lib/Support
     - `10`
     - `10`
     - `0`
     - :good:`100%`
   * - polly/lib/Transform
     - `14`
     - `14`
     - `0`
     - :good:`100%`
   * - polly/tools/GPURuntime
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - polly/unittests/DeLICM
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - polly/unittests/Flatten
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - polly/unittests/Isl
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - polly/unittests/ScheduleOptimizer
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - polly/unittests/ScopPassManager
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - polly/unittests/Support
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - pstl/include/pstl/internal
     - `22`
     - `18`
     - `4`
     - :part:`81%`
   * - Total
     - :total:`13993`
     - :total:`6737`
     - :total:`7256`
     - :total:`48%`
