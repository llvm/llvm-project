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
tree in terms of conformance to :doc:`ClangFormat` as of: June 04, 2021 13:01:37 (`e42ee2d50963 <https://github.com/llvm/llvm-project/commit/e42ee2d50963>`_).


.. list-table:: LLVM Clang-Format Status
   :widths: 50 25 25 25 25
   :header-rows: 1

   * - Directory
     - Total Files
     - Formatted Files
     - Unformatted Files
     - % Complete
   * - .
     - `1`
     - `1`
     - `0`
     - :good:`100%`
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
     - `1`
     - `0`
     - :good:`100%`
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
     - `16`
     - `4`
     - `12`
     - :part:`25%`
   * - clang/include/clang/Analysis/Analyses
     - `15`
     - `3`
     - `12`
     - :part:`20%`
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
     - `1`
     - `4`
     - :part:`20%`
   * - clang/include/clang/ASTMatchers/Dynamic
     - `4`
     - `1`
     - `3`
     - :part:`25%`
   * - clang/include/clang/Basic
     - `80`
     - `30`
     - `50`
     - :part:`37%`
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
   * - clang/include/clang/Interpreter
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - clang/include/clang/Lex
     - `29`
     - `5`
     - `24`
     - :part:`17%`
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
     - `3`
     - `11`
     - :part:`21%`
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
     - `17`
     - `10`
     - `7`
     - :part:`58%`
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
     - `1`
     - `1`
     - :part:`50%`
   * - clang/include/clang/Tooling/Refactoring
     - `15`
     - `12`
     - `3`
     - :part:`80%`
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
     - `6`
     - `2`
     - :part:`75%`
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
     - `28`
     - `4`
     - `24`
     - :part:`14%`
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
     - `3`
     - `3`
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
     - `18`
     - `26`
     - :part:`40%`
   * - clang/lib/ASTMatchers
     - `3`
     - `1`
     - `2`
     - :part:`33%`
   * - clang/lib/ASTMatchers/Dynamic
     - `6`
     - `1`
     - `5`
     - :part:`16%`
   * - clang/lib/Basic
     - `37`
     - `12`
     - `25`
     - :part:`32%`
   * - clang/lib/Basic/Targets
     - `50`
     - `24`
     - `26`
     - :part:`48%`
   * - clang/lib/CodeGen
     - `91`
     - `12`
     - `79`
     - :part:`13%`
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
     - `87`
     - `31`
     - `56`
     - :part:`35%`
   * - clang/lib/Driver/ToolChains/Arch
     - `20`
     - `6`
     - `14`
     - :part:`30%`
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
     - `32`
     - `3`
     - `29`
     - :part:`9%`
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
     - `11`
     - `1`
     - `10`
     - :part:`9%`
   * - clang/lib/IndexSerialization
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - clang/lib/Interpreter
     - `5`
     - `5`
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
     - `10`
     - `36`
     - :part:`21%`
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
     - `16`
     - `7`
     - `9`
     - :part:`43%`
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
   * - clang/lib/Tooling/DumpTool
     - `4`
     - `3`
     - `1`
     - :part:`75%`
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
     - `6`
     - `1`
     - :part:`85%`
   * - clang/lib/Tooling/Transformer
     - `7`
     - `3`
     - `4`
     - :part:`42%`
   * - clang/tools/amdgpu-arch
     - `1`
     - `1`
     - `0`
     - :good:`100%`
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
   * - clang/tools/clang-repl
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - clang/tools/clang-scan-deps
     - `1`
     - `1`
     - `0`
     - :good:`100%`
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
     - `6`
     - `2`
     - `4`
     - :part:`33%`
   * - clang/unittests/AST
     - `28`
     - `6`
     - `22`
     - :part:`21%`
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
     - `7`
     - `3`
     - `4`
     - :part:`42%`
   * - clang/unittests/CodeGen
     - `6`
     - `1`
     - `5`
     - :part:`16%`
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
   * - clang/unittests/Interpreter
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - clang/unittests/Introspection
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - clang/unittests/Lex
     - `7`
     - `3`
     - `4`
     - :part:`42%`
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
     - `4`
     - `8`
     - :part:`33%`
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
     - `3`
     - `4`
     - :part:`42%`
   * - clang/utils/perf-training/cxx
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - clang/utils/TableGen
     - `22`
     - `4`
     - `18`
     - :part:`18%`
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
     - `0`
     - `1`
     - :none:`0%`
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
     - `11`
     - `9`
     - `2`
     - :part:`81%`
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
     - `115`
     - `94`
     - `21`
     - :part:`81%`
   * - clang-tools-extra/clang-tidy/cert
     - `29`
     - `28`
     - `1`
     - :part:`96%`
   * - clang-tools-extra/clang-tidy/concurrency
     - `5`
     - `4`
     - `1`
     - :part:`80%`
   * - clang-tools-extra/clang-tidy/cppcoreguidelines
     - `43`
     - `40`
     - `3`
     - :part:`93%`
   * - clang-tools-extra/clang-tidy/darwin
     - `5`
     - `2`
     - `3`
     - :part:`40%`
   * - clang-tools-extra/clang-tidy/fuchsia
     - `15`
     - `10`
     - `5`
     - :part:`66%`
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
     - `49`
     - `18`
     - :part:`73%`
   * - clang-tools-extra/clang-tidy/mpi
     - `5`
     - `5`
     - `0`
     - :good:`100%`
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
     - `31`
     - `24`
     - `7`
     - :part:`77%`
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
     - `77`
     - `63`
     - `14`
     - :part:`81%`
   * - clang-tools-extra/clang-tidy/tool
     - `3`
     - `2`
     - `1`
     - :part:`66%`
   * - clang-tools-extra/clang-tidy/utils
     - `35`
     - `30`
     - `5`
     - :part:`85%`
   * - clang-tools-extra/clang-tidy/zircon
     - `3`
     - `3`
     - `0`
     - :good:`100%`
   * - clang-tools-extra/clangd
     - `93`
     - `79`
     - `14`
     - :part:`84%`
   * - clang-tools-extra/clangd/benchmarks
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - clang-tools-extra/clangd/benchmarks/CompletionModel
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - clang-tools-extra/clangd/fuzzer
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - clang-tools-extra/clangd/index
     - `39`
     - `37`
     - `2`
     - :part:`94%`
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
   * - clang-tools-extra/clangd/index/remote/monitor
     - `1`
     - `1`
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
     - `3`
     - `1`
     - :part:`75%`
   * - clang-tools-extra/clangd/refactor/tweaks
     - `14`
     - `11`
     - `3`
     - :part:`78%`
   * - clang-tools-extra/clangd/support
     - `23`
     - `22`
     - `1`
     - :part:`95%`
   * - clang-tools-extra/clangd/tool
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - clang-tools-extra/clangd/unittests
     - `77`
     - `66`
     - `11`
     - :part:`85%`
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
     - `11`
     - `11`
     - `0`
     - :good:`100%`
   * - clang-tools-extra/clangd/unittests/tweaks
     - `20`
     - `20`
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
     - `6`
     - `9`
     - :part:`40%`
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
     - `13`
     - `8`
     - `5`
     - :part:`61%`
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
     - `12`
     - `12`
     - `0`
     - :good:`100%`
   * - compiler-rt/lib/gwp_asan/optional
     - `10`
     - `10`
     - `0`
     - :good:`100%`
   * - compiler-rt/lib/gwp_asan/platform_specific
     - `13`
     - `13`
     - `0`
     - :good:`100%`
   * - compiler-rt/lib/gwp_asan/tests
     - `15`
     - `15`
     - `0`
     - :good:`100%`
   * - compiler-rt/lib/gwp_asan/tests/platform_specific
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
     - `2`
     - `18`
     - :part:`10%`
   * - compiler-rt/lib/memprof
     - `27`
     - `26`
     - `1`
     - :part:`96%`
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
   * - compiler-rt/lib/orc
     - `10`
     - `8`
     - `2`
     - :part:`80%`
   * - compiler-rt/lib/orc/unittests
     - `8`
     - `7`
     - `1`
     - :part:`87%`
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
     - `162`
     - `18`
     - `144`
     - :part:`11%`
   * - compiler-rt/lib/sanitizer_common/symbolizer
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - compiler-rt/lib/sanitizer_common/tests
     - `39`
     - `2`
     - `37`
     - :part:`5%`
   * - compiler-rt/lib/scudo
     - `20`
     - `0`
     - `20`
     - :none:`0%`
   * - compiler-rt/lib/scudo/standalone
     - `47`
     - `47`
     - `0`
     - :good:`100%`
   * - compiler-rt/lib/scudo/standalone/benchmarks
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - compiler-rt/lib/scudo/standalone/fuzz
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - compiler-rt/lib/scudo/standalone/include/scudo
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - compiler-rt/lib/scudo/standalone/tests
     - `24`
     - `24`
     - `0`
     - :good:`100%`
   * - compiler-rt/lib/scudo/standalone/tools
     - `1`
     - `1`
     - `0`
     - :good:`100%`
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
     - `0`
     - `10`
     - :none:`0%`
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
     - `27`
     - `12`
     - :part:`69%`
   * - compiler-rt/lib/xray/tests/unit
     - `10`
     - `8`
     - `2`
     - :part:`80%`
   * - compiler-rt/tools/gwp_asan
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - debuginfo-tests/dexter/feature_tests/commands/penalty
     - `7`
     - `0`
     - `7`
     - :none:`0%`
   * - debuginfo-tests/dexter/feature_tests/commands/perfect
     - `5`
     - `0`
     - `5`
     - :none:`0%`
   * - debuginfo-tests/dexter/feature_tests/commands/perfect/dex_declare_file/dex_and_source
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - debuginfo-tests/dexter/feature_tests/commands/perfect/dex_declare_file/precompiled_binary
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - debuginfo-tests/dexter/feature_tests/commands/perfect/dex_declare_file/precompiled_binary_different_dir/source
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - debuginfo-tests/dexter/feature_tests/commands/perfect/dex_declare_file/windows_noncanonical_path/source
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - debuginfo-tests/dexter/feature_tests/commands/perfect/expect_step_kind
     - `5`
     - `0`
     - `5`
     - :none:`0%`
   * - debuginfo-tests/dexter/feature_tests/commands/perfect/limit_steps
     - `8`
     - `2`
     - `6`
     - :part:`25%`
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
     - `15`
     - `3`
     - `12`
     - :part:`20%`
   * - debuginfo-tests/llgdb-tests
     - `8`
     - `0`
     - `8`
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
     - `20`
     - `20`
     - `0`
     - :good:`100%`
   * - flang/include/flang/Decimal
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - flang/include/flang/Evaluate
     - `23`
     - `23`
     - `0`
     - :good:`100%`
   * - flang/include/flang/Frontend
     - `9`
     - `9`
     - `0`
     - :good:`100%`
   * - flang/include/flang/FrontendTool
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - flang/include/flang/Lower
     - `19`
     - `19`
     - `0`
     - :good:`100%`
   * - flang/include/flang/Lower/Support
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - flang/include/flang/Optimizer/CodeGen
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - flang/include/flang/Optimizer/Dialect
     - `5`
     - `5`
     - `0`
     - :good:`100%`
   * - flang/include/flang/Optimizer/Support
     - `5`
     - `4`
     - `1`
     - :part:`80%`
   * - flang/include/flang/Optimizer/Transforms
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - flang/include/flang/Parser
     - `17`
     - `16`
     - `1`
     - :part:`94%`
   * - flang/include/flang/Semantics
     - `9`
     - `8`
     - `1`
     - :part:`88%`
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
     - `30`
     - `1`
     - :part:`96%`
   * - flang/lib/Frontend
     - `8`
     - `8`
     - `0`
     - :good:`100%`
   * - flang/lib/FrontendTool
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - flang/lib/Lower
     - `17`
     - `17`
     - `0`
     - :good:`100%`
   * - flang/lib/Optimizer/CodeGen
     - `4`
     - `3`
     - `1`
     - :part:`75%`
   * - flang/lib/Optimizer/Dialect
     - `4`
     - `2`
     - `2`
     - :part:`50%`
   * - flang/lib/Optimizer/Support
     - `3`
     - `3`
     - `0`
     - :good:`100%`
   * - flang/lib/Optimizer/Transforms
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - flang/lib/Parser
     - `35`
     - `35`
     - `0`
     - :good:`100%`
   * - flang/lib/Semantics
     - `78`
     - `72`
     - `6`
     - :part:`92%`
   * - flang/module
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - flang/runtime
     - `82`
     - `82`
     - `0`
     - :good:`100%`
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
   * - flang/tools/fir-opt
     - `1`
     - `1`
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
     - `15`
     - `0`
     - :good:`100%`
   * - flang/unittests/Frontend
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - flang/unittests/Optimizer
     - `3`
     - `2`
     - `1`
     - :part:`66%`
   * - flang/unittests/Runtime
     - `5`
     - `5`
     - `0`
     - :good:`100%`
   * - flang/unittests/RuntimeGTest
     - `15`
     - `14`
     - `1`
     - :part:`93%`
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
     - `12`
     - `12`
     - `0`
     - :good:`100%`
   * - libc/config/linux
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - libc/fuzzing/math
     - `6`
     - `6`
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
     - `33`
     - `33`
     - `0`
     - :good:`100%`
   * - libc/src/errno
     - `3`
     - `3`
     - `0`
     - :good:`100%`
   * - libc/src/fenv
     - `22`
     - `22`
     - `0`
     - :good:`100%`
   * - libc/src/math
     - `86`
     - `86`
     - `0`
     - :good:`100%`
   * - libc/src/math/aarch64
     - `10`
     - `10`
     - `0`
     - :good:`100%`
   * - libc/src/math/generic
     - `85`
     - `85`
     - `0`
     - :good:`100%`
   * - libc/src/math/x86_64
     - `3`
     - `3`
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
     - `9`
     - `9`
     - `0`
     - :good:`100%`
   * - libc/src/stdlib/linux
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - libc/src/string
     - `43`
     - `43`
     - `0`
     - :good:`100%`
   * - libc/src/string/aarch64
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - libc/src/string/memory_utils
     - `3`
     - `3`
     - `0`
     - :good:`100%`
   * - libc/src/string/x86_64
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
     - `12`
     - `12`
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
   * - libc/src/__support
     - `4`
     - `4`
     - `0`
     - :good:`100%`
   * - libc/utils/CPP
     - `6`
     - `6`
     - `0`
     - :good:`100%`
   * - libc/utils/FPUtil
     - `20`
     - `20`
     - `0`
     - :good:`100%`
   * - libc/utils/FPUtil/aarch64
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - libc/utils/FPUtil/generic
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - libc/utils/FPUtil/x86_64
     - `2`
     - `2`
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
     - `10`
     - `9`
     - `1`
     - :part:`90%`
   * - libc/utils/tools/WrapperGen
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - libc/utils/UnitTest
     - `4`
     - `4`
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
     - `22`
     - `5`
     - `17`
     - :part:`22%`
   * - libcxx/include
     - `22`
     - `5`
     - `17`
     - :part:`22%`
   * - libcxx/include/__format
     - `2`
     - `0`
     - `2`
     - :none:`0%`
   * - libcxx/include/__iterator
     - `11`
     - `3`
     - `8`
     - :part:`27%`
   * - libcxx/include/__memory
     - `14`
     - `0`
     - `14`
     - :none:`0%`
   * - libcxx/include/__ranges
     - `8`
     - `1`
     - `7`
     - :part:`12%`
   * - libcxx/include/__support/android
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - libcxx/include/__support/fuchsia
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - libcxx/include/__support/ibm
     - `7`
     - `2`
     - `5`
     - :part:`28%`
   * - libcxx/include/__support/musl
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - libcxx/include/__support/newlib
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - libcxx/include/__support/nuttx
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - libcxx/include/__support/openbsd
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - libcxx/include/__support/solaris
     - `3`
     - `2`
     - `1`
     - :part:`66%`
   * - libcxx/include/__support/win32
     - `2`
     - `0`
     - `2`
     - :none:`0%`
   * - libcxx/include/__support/xlocale
     - `3`
     - `0`
     - `3`
     - :none:`0%`
   * - libcxx/include/__utility
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - libcxx/src
     - `38`
     - `3`
     - `35`
     - :part:`7%`
   * - libcxx/src/experimental
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - libcxx/src/filesystem
     - `5`
     - `0`
     - `5`
     - :none:`0%`
   * - libcxx/src/include
     - `5`
     - `2`
     - `3`
     - :part:`40%`
   * - libcxx/src/support/ibm
     - `1`
     - `0`
     - `1`
     - :none:`0%`
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
     - `24`
     - `24`
     - :part:`50%`
   * - lld/ELF/Arch
     - `14`
     - `5`
     - `9`
     - :part:`35%`
   * - lld/include/lld/Common
     - `13`
     - `6`
     - `7`
     - :part:`46%`
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
     - `41`
     - `38`
     - `3`
     - :part:`92%`
   * - lld/MachO/Arch
     - `6`
     - `6`
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
     - `17`
     - `12`
     - :part:`58%`
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
     - `10`
     - `15`
     - :part:`40%`
   * - lldb/include/lldb/Core
     - `60`
     - `32`
     - `28`
     - :part:`53%`
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
     - `21`
     - `19`
     - :part:`52%`
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
     - `6`
     - `4`
     - `2`
     - :part:`66%`
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
     - `10`
     - `4`
     - `6`
     - :part:`40%`
   * - lldb/include/lldb/Initialization
     - `3`
     - `1`
     - `2`
     - :part:`33%`
   * - lldb/include/lldb/Interpreter
     - `48`
     - `36`
     - `12`
     - :part:`75%`
   * - lldb/include/lldb/Symbol
     - `35`
     - `16`
     - `19`
     - :part:`45%`
   * - lldb/include/lldb/Target
     - `72`
     - `44`
     - `28`
     - :part:`61%`
   * - lldb/include/lldb/Utility
     - `64`
     - `41`
     - `23`
     - :part:`64%`
   * - lldb/source
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - lldb/source/API
     - `75`
     - `6`
     - `69`
     - :part:`8%`
   * - lldb/source/Breakpoint
     - `24`
     - `5`
     - `19`
     - :part:`20%`
   * - lldb/source/Commands
     - `68`
     - `60`
     - `8`
     - :part:`88%`
   * - lldb/source/Core
     - `48`
     - `25`
     - `23`
     - :part:`52%`
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
     - `16`
     - `16`
     - :part:`50%`
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
     - `11`
     - `5`
     - `6`
     - :part:`45%`
   * - lldb/source/Initialization
     - `3`
     - `3`
     - `0`
     - :good:`100%`
   * - lldb/source/Interpreter
     - `44`
     - `22`
     - `22`
     - :part:`50%`
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
     - `2`
     - `4`
     - :part:`33%`
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
     - `3`
     - `8`
     - :part:`27%`
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
     - `3`
     - `1`
     - :part:`75%`
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
     - `14`
     - `6`
     - :part:`70%`
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
     - `0`
     - `2`
     - :none:`0%`
   * - lldb/source/Plugins/LanguageRuntime/ObjC/AppleObjCRuntime
     - `16`
     - `6`
     - `10`
     - :part:`37%`
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
     - `1`
     - `1`
     - :part:`50%`
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
     - `16`
     - `12`
     - `4`
     - :part:`75%`
   * - lldb/source/Plugins/Process/gdb-remote
     - `26`
     - `16`
     - `10`
     - :part:`61%`
   * - lldb/source/Plugins/Process/Linux
     - `21`
     - `11`
     - `10`
     - :part:`52%`
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
     - `132`
     - `95`
     - `37`
     - :part:`71%`
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
     - `12`
     - `7`
     - `5`
     - :part:`58%`
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
     - `37`
     - `28`
     - :part:`56%`
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
     - `11`
     - `11`
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
     - `31`
     - `17`
     - `14`
     - :part:`54%`
   * - lldb/source/Target
     - `65`
     - `32`
     - `33`
     - :part:`49%`
   * - lldb/source/Utility
     - `58`
     - `45`
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
     - `40`
     - `9`
     - :part:`81%`
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
     - `2`
     - `3`
     - :part:`40%`
   * - lldb/tools/lldb-vscode
     - `27`
     - `26`
     - `1`
     - :part:`96%`
   * - lldb/unittests
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - lldb/unittests/API
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - lldb/unittests/Breakpoint
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - lldb/unittests/Core
     - `10`
     - `9`
     - `1`
     - :part:`90%`
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
     - `2`
     - `3`
     - :part:`40%`
   * - lldb/unittests/Host
     - `14`
     - `11`
     - `3`
     - :part:`78%`
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
     - `4`
     - `1`
     - `3`
     - :part:`25%`
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
     - `5`
     - `3`
     - `2`
     - :part:`60%`
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
     - `4`
     - `1`
     - `3`
     - :part:`25%`
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
     - `0`
     - `8`
     - :none:`0%`
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
     - `45`
     - `32`
     - `13`
     - :part:`71%`
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
   * - llvm/bindings/ocaml/llvm
     - `1`
     - `1`
     - `0`
     - :good:`100%`
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
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/examples/OrcV2Examples/LLJITWithOptimizingIRTransform
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - llvm/examples/OrcV2Examples/LLJITWithRemoteDebugging
     - `3`
     - `3`
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
     - `87`
     - `22`
     - `65`
     - :part:`25%`
   * - llvm/include/llvm/Analysis
     - `123`
     - `41`
     - `82`
     - :part:`33%`
   * - llvm/include/llvm/Analysis/Utils
     - `3`
     - `1`
     - `2`
     - :part:`33%`
   * - llvm/include/llvm/AsmParser
     - `5`
     - `2`
     - `3`
     - :part:`40%`
   * - llvm/include/llvm/BinaryFormat
     - `14`
     - `9`
     - `5`
     - :part:`64%`
   * - llvm/include/llvm/Bitcode
     - `7`
     - `3`
     - `4`
     - :part:`42%`
   * - llvm/include/llvm/Bitstream
     - `3`
     - `0`
     - `3`
     - :none:`0%`
   * - llvm/include/llvm/CodeGen
     - `149`
     - `44`
     - `105`
     - :part:`29%`
   * - llvm/include/llvm/CodeGen/GlobalISel
     - `27`
     - `9`
     - `18`
     - :part:`33%`
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
     - `3`
     - `11`
     - :part:`21%`
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
     - `1`
     - `2`
     - :part:`33%`
   * - llvm/include/llvm/Demangle
     - `8`
     - `3`
     - `5`
     - :part:`37%`
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
     - `10`
     - `7`
     - `3`
     - :part:`70%`
   * - llvm/include/llvm/ExecutionEngine/Orc
     - `32`
     - `21`
     - `11`
     - :part:`65%`
   * - llvm/include/llvm/ExecutionEngine/Orc/Shared
     - `6`
     - `5`
     - `1`
     - :part:`83%`
   * - llvm/include/llvm/ExecutionEngine/Orc/TargetProcess
     - `4`
     - `4`
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
     - `92`
     - `25`
     - `67`
     - :part:`27%`
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
     - `70`
     - `19`
     - `51`
     - :part:`27%`
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
     - `8`
     - `7`
     - `1`
     - :part:`87%`
   * - llvm/include/llvm/Object
     - `31`
     - `11`
     - `20`
     - :part:`35%`
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
     - `3`
     - `5`
     - :part:`37%`
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
     - `176`
     - `57`
     - `119`
     - :part:`32%`
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
     - `6`
     - `2`
     - `4`
     - :part:`33%`
   * - llvm/include/llvm/Testing/Support
     - `3`
     - `2`
     - `1`
     - :part:`66%`
   * - llvm/include/llvm/TextAPI
     - `9`
     - `9`
     - `0`
     - :good:`100%`
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
   * - llvm/include/llvm/Transforms/InstCombine
     - `3`
     - `1`
     - `2`
     - :part:`33%`
   * - llvm/include/llvm/Transforms/Instrumentation
     - `17`
     - `11`
     - `6`
     - :part:`64%`
   * - llvm/include/llvm/Transforms/IPO
     - `37`
     - `26`
     - `11`
     - :part:`70%`
   * - llvm/include/llvm/Transforms/Scalar
     - `73`
     - `45`
     - `28`
     - :part:`61%`
   * - llvm/include/llvm/Transforms/Utils
     - `70`
     - `41`
     - `29`
     - :part:`58%`
   * - llvm/include/llvm/Transforms/Vectorize
     - `5`
     - `1`
     - `4`
     - :part:`20%`
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
     - `13`
     - `4`
     - :part:`76%`
   * - llvm/include/llvm-c
     - `26`
     - `12`
     - `14`
     - :part:`46%`
   * - llvm/include/llvm-c/Transforms
     - `9`
     - `3`
     - `6`
     - :part:`33%`
   * - llvm/lib/Analysis
     - `117`
     - `38`
     - `79`
     - :part:`32%`
   * - llvm/lib/AsmParser
     - `3`
     - `1`
     - `2`
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
     - `204`
     - `46`
     - `158`
     - :part:`22%`
   * - llvm/lib/CodeGen/AsmPrinter
     - `45`
     - `16`
     - `29`
     - :part:`35%`
   * - llvm/lib/CodeGen/GlobalISel
     - `25`
     - `8`
     - `17`
     - :part:`32%`
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
     - `2`
     - `9`
     - :part:`18%`
   * - llvm/lib/DebugInfo/MSF
     - `4`
     - `3`
     - `1`
     - :part:`75%`
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
     - `5`
     - `3`
     - `2`
     - :part:`60%`
   * - llvm/lib/DWARFLinker
     - `4`
     - `4`
     - `0`
     - :good:`100%`
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
     - `16`
     - `8`
     - `8`
     - :part:`50%`
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
     - `28`
     - `17`
     - `11`
     - :part:`60%`
   * - llvm/lib/ExecutionEngine/Orc/Shared
     - `3`
     - `3`
     - `0`
     - :good:`100%`
   * - llvm/lib/ExecutionEngine/Orc/TargetProcess
     - `3`
     - `3`
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
   * - llvm/lib/Frontend/OpenACC
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - llvm/lib/Frontend/OpenMP
     - `3`
     - `2`
     - `1`
     - :part:`66%`
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
     - `68`
     - `17`
     - `51`
     - :part:`25%`
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
     - `63`
     - `21`
     - `42`
     - :part:`33%`
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
     - `6`
     - `1`
     - :part:`85%`
   * - llvm/lib/MCA/HardwareUnits
     - `6`
     - `3`
     - `3`
     - :part:`50%`
   * - llvm/lib/MCA/Stages
     - `8`
     - `8`
     - `0`
     - :good:`100%`
   * - llvm/lib/Object
     - `31`
     - `15`
     - `16`
     - :part:`48%`
   * - llvm/lib/ObjectYAML
     - `22`
     - `10`
     - `12`
     - :part:`45%`
   * - llvm/lib/Option
     - `4`
     - `0`
     - `4`
     - :none:`0%`
   * - llvm/lib/Passes
     - `4`
     - `2`
     - `2`
     - :part:`50%`
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
     - `136`
     - `52`
     - `84`
     - :part:`38%`
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
     - `59`
     - `5`
     - `54`
     - :part:`8%`
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
     - `14`
     - `4`
     - `10`
     - :part:`28%`
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
     - `149`
     - `21`
     - `128`
     - :part:`14%`
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
     - `11`
     - `4`
     - `7`
     - :part:`36%`
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
     - `75`
     - `8`
     - `67`
     - :part:`10%`
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
     - `31`
     - `8`
     - `23`
     - :part:`25%`
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
   * - llvm/lib/Target/CSKY/AsmParser
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - llvm/lib/Target/CSKY/MCTargetDesc
     - `14`
     - `14`
     - `0`
     - :good:`100%`
   * - llvm/lib/Target/CSKY/TargetInfo
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - llvm/lib/Target/Hexagon
     - `80`
     - `4`
     - `76`
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
   * - llvm/lib/Target/M68k
     - `26`
     - `25`
     - `1`
     - :part:`96%`
   * - llvm/lib/Target/M68k/AsmParser
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - llvm/lib/Target/M68k/Disassembler
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - llvm/lib/Target/M68k/MCTargetDesc
     - `12`
     - `12`
     - `0`
     - :good:`100%`
   * - llvm/lib/Target/M68k/TargetInfo
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - llvm/lib/Target/Mips
     - `69`
     - `11`
     - `58`
     - :part:`15%`
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
     - `44`
     - `9`
     - `35`
     - :part:`20%`
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
     - `3`
     - `49`
     - :part:`5%`
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
     - `20`
     - `5`
     - `15`
     - :part:`25%`
   * - llvm/lib/Target/PowerPC/TargetInfo
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - llvm/lib/Target/RISCV
     - `33`
     - `18`
     - `15`
     - :part:`54%`
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
     - `21`
     - `11`
     - `10`
     - :part:`52%`
   * - llvm/lib/Target/RISCV/TargetInfo
     - `2`
     - `2`
     - `0`
     - :good:`100%`
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
     - `5`
     - `35`
     - :part:`12%`
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
     - `21`
     - `17`
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
     - `59`
     - `43`
     - `16`
     - :part:`72%`
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
     - `8`
     - `4`
     - :part:`66%`
   * - llvm/lib/Target/WebAssembly/TargetInfo
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - llvm/lib/Target/WebAssembly/Utils
     - `4`
     - `4`
     - `0`
     - :good:`100%`
   * - llvm/lib/Target/X86
     - `82`
     - `18`
     - `64`
     - :part:`21%`
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
     - `11`
     - `0`
     - :good:`100%`
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
   * - llvm/lib/Transforms/InstCombine
     - `16`
     - `1`
     - `15`
     - :part:`6%`
   * - llvm/lib/Transforms/Instrumentation
     - `22`
     - `6`
     - `16`
     - :part:`27%`
   * - llvm/lib/Transforms/IPO
     - `42`
     - `9`
     - `33`
     - :part:`21%`
   * - llvm/lib/Transforms/ObjCARC
     - `15`
     - `4`
     - `11`
     - :part:`26%`
   * - llvm/lib/Transforms/Scalar
     - `78`
     - `13`
     - `65`
     - :part:`16%`
   * - llvm/lib/Transforms/Utils
     - `75`
     - `17`
     - `58`
     - :part:`22%`
   * - llvm/lib/Transforms/Vectorize
     - `22`
     - `12`
     - `10`
     - :part:`54%`
   * - llvm/lib/WindowsManifest
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - llvm/lib/XRay
     - `14`
     - `11`
     - `3`
     - :part:`78%`
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
     - `4`
     - `2`
     - `2`
     - :part:`50%`
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
     - `0`
     - `1`
     - :none:`0%`
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
     - `1`
     - `0`
     - :good:`100%`
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
     - `1`
     - `0`
     - :good:`100%`
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
     - `22`
     - `13`
     - `9`
     - :part:`59%`
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
     - `9`
     - `9`
     - `0`
     - :good:`100%`
   * - llvm/tools/llvm-objcopy/ELF
     - `5`
     - `3`
     - `2`
     - :part:`60%`
   * - llvm/tools/llvm-objcopy/MachO
     - `11`
     - `11`
     - `0`
     - :good:`100%`
   * - llvm/tools/llvm-objcopy/wasm
     - `9`
     - `9`
     - `0`
     - :good:`100%`
   * - llvm/tools/llvm-objdump
     - `15`
     - `10`
     - `5`
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
     - `15`
     - `32`
     - :part:`31%`
   * - llvm/tools/llvm-profdata
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/tools/llvm-profgen
     - `13`
     - `12`
     - `1`
     - :part:`92%`
   * - llvm/tools/llvm-rc
     - `12`
     - `6`
     - `6`
     - :part:`50%`
   * - llvm/tools/llvm-readobj
     - `19`
     - `4`
     - `15`
     - :part:`21%`
   * - llvm/tools/llvm-reduce
     - `5`
     - `4`
     - `1`
     - :part:`80%`
   * - llvm/tools/llvm-reduce/deltas
     - `30`
     - `28`
     - `2`
     - :part:`93%`
   * - llvm/tools/llvm-rtdyld
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - llvm/tools/llvm-rust-demangle-fuzzer
     - `2`
     - `2`
     - `0`
     - :good:`100%`
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
     - `1`
     - `0`
     - :good:`100%`
   * - llvm/tools/llvm-undname
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - llvm/tools/llvm-xray
     - `19`
     - `15`
     - `4`
     - :part:`78%`
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
     - `77`
     - `31`
     - `46`
     - :part:`40%`
   * - llvm/unittests/Analysis
     - `36`
     - `10`
     - `26`
     - :part:`27%`
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
     - `12`
     - `2`
     - `10`
     - :part:`16%`
   * - llvm/unittests/DebugInfo/CodeView
     - `3`
     - `1`
     - `2`
     - :part:`33%`
   * - llvm/unittests/DebugInfo/DWARF
     - `16`
     - `10`
     - `6`
     - :part:`62%`
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
     - `5`
     - `3`
     - `2`
     - :part:`60%`
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
     - `15`
     - `6`
     - `9`
     - :part:`40%`
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
   * - llvm/unittests/MC/SystemZ
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
     - `4`
     - `4`
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
     - `95`
     - `30`
     - `65`
     - :part:`31%`
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
   * - llvm/unittests/tools/llvm-profgen
     - `1`
     - `0`
     - `1`
     - :none:`0%`
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
     - `19`
     - `8`
     - `11`
     - :part:`42%`
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
     - `76`
     - `10`
     - `66`
     - :part:`13%`
   * - llvm/utils/TableGen/GlobalISel
     - `17`
     - `8`
     - `9`
     - :part:`47%`
   * - llvm/utils/unittest/googlemock/include/gmock
     - `12`
     - `0`
     - `12`
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
     - `11`
     - `0`
     - `11`
     - :none:`0%`
   * - llvm/utils/unittest/googletest/include/gtest/internal
     - `8`
     - `0`
     - `8`
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
     - `5`
     - `0`
     - :good:`100%`
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
     - `6`
     - `0`
     - :good:`100%`
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
     - `6`
     - `0`
     - :good:`100%`
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
     - `14`
     - `12`
     - `2`
     - :part:`85%`
   * - mlir/include/mlir/Analysis/AliasAnalysis
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Analysis/Presburger
     - `3`
     - `3`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Bindings/Python
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/CAPI
     - `11`
     - `11`
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
   * - mlir/include/mlir/Conversion/ComplexToLLVM
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Conversion/ComplexToStandard
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
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Conversion/GPUToROCDL
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Conversion/GPUToSPIRV
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Conversion/GPUToVulkan
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - mlir/include/mlir/Conversion/LinalgToLLVM
     - `1`
     - `1`
     - `0`
     - :good:`100%`
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
   * - mlir/include/mlir/Conversion/MathToLibm
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Conversion/OpenACCToLLVM
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Conversion/OpenMPToLLVM
     - `1`
     - `1`
     - `0`
     - :good:`100%`
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
     - `2`
     - `2`
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
   * - mlir/include/mlir/Conversion/TosaToLinalg
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Conversion/TosaToSCF
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Conversion/TosaToStandard
     - `1`
     - `1`
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
   * - mlir/include/mlir/Dialect/Affine/IR
     - `3`
     - `3`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Dialect/AMX
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Dialect/ArmNeon
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Dialect/ArmSVE
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Dialect/Async
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Dialect/Async/IR
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Dialect/Complex/IR
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Dialect/DLTI
     - `2`
     - `2`
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
   * - mlir/include/mlir/Dialect/Linalg/IR
     - `3`
     - `3`
     - `0`
     - :good:`100%`
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
   * - mlir/include/mlir/Dialect/Math/IR
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Dialect/Math/Transforms
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Dialect/MemRef/IR
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Dialect/MemRef/Transforms
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Dialect/MemRef/Utils
     - `1`
     - `1`
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
     - `3`
     - `3`
     - `0`
     - :good:`100%`
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
     - `2`
     - `2`
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
   * - mlir/include/mlir/Dialect/SparseTensor/IR
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Dialect/SparseTensor/Transforms
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Dialect/SPIRV/IR
     - `9`
     - `9`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Dialect/SPIRV/Linking
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Dialect/SPIRV/Transforms
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Dialect/SPIRV/Utils
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Dialect/StandardOps/IR
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Dialect/StandardOps/Transforms
     - `4`
     - `4`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Dialect/StandardOps/Utils
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Dialect/Tensor/IR
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Dialect/Tensor/Transforms
     - `1`
     - `1`
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
   * - mlir/include/mlir/Dialect/X86Vector
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/ExecutionEngine
     - `7`
     - `5`
     - `2`
     - :part:`71%`
   * - mlir/include/mlir/Interfaces
     - `13`
     - `12`
     - `1`
     - :part:`92%`
   * - mlir/include/mlir/IR
     - `47`
     - `25`
     - `22`
     - :part:`53%`
   * - mlir/include/mlir/Parser
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Pass
     - `6`
     - `0`
     - `6`
     - :none:`0%`
   * - mlir/include/mlir/Reducer
     - `5`
     - `5`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Rewrite
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Support
     - `15`
     - `8`
     - `7`
     - :part:`53%`
   * - mlir/include/mlir/TableGen
     - `21`
     - `20`
     - `1`
     - :part:`95%`
   * - mlir/include/mlir/Target/LLVMIR
     - `5`
     - `4`
     - `1`
     - :part:`80%`
   * - mlir/include/mlir/Target/LLVMIR/Dialect
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Target/LLVMIR/Dialect/AMX
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Target/LLVMIR/Dialect/ArmNeon
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Target/LLVMIR/Dialect/ArmSVE
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Target/LLVMIR/Dialect/LLVMIR
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Target/LLVMIR/Dialect/NVVM
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Target/LLVMIR/Dialect/OpenACC
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Target/LLVMIR/Dialect/OpenMP
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Target/LLVMIR/Dialect/ROCDL
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Target/LLVMIR/Dialect/X86Vector
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Target/SPIRV
     - `3`
     - `3`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Tools/mlir-lsp-server
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Tools/mlir-reduce
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir/Transforms
     - `14`
     - `9`
     - `5`
     - :part:`64%`
   * - mlir/include/mlir-c
     - `14`
     - `14`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir-c/Bindings/Python
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/include/mlir-c/Dialect
     - `8`
     - `8`
     - `0`
     - :good:`100%`
   * - mlir/lib/Analysis
     - `14`
     - `13`
     - `1`
     - :part:`92%`
   * - mlir/lib/Analysis/AliasAnalysis
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/lib/Analysis/Presburger
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - mlir/lib/Bindings/Python
     - `22`
     - `22`
     - `0`
     - :good:`100%`
   * - mlir/lib/Bindings/Python/Conversions
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/lib/Bindings/Python/Transforms
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/lib/CAPI/Conversion
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/lib/CAPI/Debug
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/lib/CAPI/Dialect
     - `12`
     - `12`
     - `0`
     - :good:`100%`
   * - mlir/lib/CAPI/ExecutionEngine
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/lib/CAPI/IR
     - `10`
     - `10`
     - `0`
     - :good:`100%`
   * - mlir/lib/CAPI/Registration
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
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/lib/Conversion/AsyncToLLVM
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/lib/Conversion/ComplexToLLVM
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/lib/Conversion/ComplexToStandard
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
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - mlir/lib/Conversion/GPUToROCDL
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/lib/Conversion/GPUToSPIRV
     - `2`
     - `2`
     - `0`
     - :good:`100%`
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
   * - mlir/lib/Conversion/MathToLibm
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/lib/Conversion/OpenACCToLLVM
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
     - `5`
     - `0`
     - :good:`100%`
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
     - `2`
     - `2`
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
     - `3`
     - `0`
     - :good:`100%`
   * - mlir/lib/Conversion/StandardToLLVM
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - mlir/lib/Conversion/StandardToSPIRV
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - mlir/lib/Conversion/TosaToLinalg
     - `2`
     - `0`
     - `2`
     - :none:`0%`
   * - mlir/lib/Conversion/TosaToSCF
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - mlir/lib/Conversion/TosaToStandard
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - mlir/lib/Conversion/VectorToLLVM
     - `2`
     - `2`
     - `0`
     - :good:`100%`
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
     - `2`
     - `1`
     - `1`
     - :part:`50%`
   * - mlir/lib/Dialect
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
   * - mlir/lib/Dialect/AMX/IR
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/lib/Dialect/AMX/Transforms
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/lib/Dialect/ArmNeon/IR
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/lib/Dialect/ArmSVE/IR
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/lib/Dialect/ArmSVE/Transforms
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
     - `5`
     - `5`
     - `0`
     - :good:`100%`
   * - mlir/lib/Dialect/Complex/IR
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - mlir/lib/Dialect/DLTI
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - mlir/lib/Dialect/GPU/IR
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/lib/Dialect/GPU/Transforms
     - `9`
     - `9`
     - `0`
     - :good:`100%`
   * - mlir/lib/Dialect/Linalg/Analysis
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/lib/Dialect/Linalg/IR
     - `3`
     - `3`
     - `0`
     - :good:`100%`
   * - mlir/lib/Dialect/Linalg/Transforms
     - `19`
     - `18`
     - `1`
     - :part:`94%`
   * - mlir/lib/Dialect/Linalg/Utils
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/lib/Dialect/LLVMIR/IR
     - `7`
     - `7`
     - `0`
     - :good:`100%`
   * - mlir/lib/Dialect/LLVMIR/Transforms
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - mlir/lib/Dialect/Math/IR
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - mlir/lib/Dialect/Math/Transforms
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - mlir/lib/Dialect/MemRef/IR
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - mlir/lib/Dialect/MemRef/Transforms
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/lib/Dialect/MemRef/Utils
     - `1`
     - `1`
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
     - `2`
     - `2`
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
   * - mlir/lib/Dialect/SparseTensor/IR
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/lib/Dialect/SparseTensor/Transforms
     - `3`
     - `3`
     - `0`
     - :good:`100%`
   * - mlir/lib/Dialect/SPIRV/IR
     - `8`
     - `6`
     - `2`
     - :part:`75%`
   * - mlir/lib/Dialect/SPIRV/Linking/ModuleCombiner
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/lib/Dialect/SPIRV/Transforms
     - `6`
     - `5`
     - `1`
     - :part:`83%`
   * - mlir/lib/Dialect/SPIRV/Utils
     - `1`
     - `1`
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
   * - mlir/lib/Dialect/StandardOps/Utils
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/lib/Dialect/Tensor/IR
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - mlir/lib/Dialect/Tensor/Transforms
     - `2`
     - `2`
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
   * - mlir/lib/Dialect/Utils
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/lib/Dialect/Vector
     - `4`
     - `3`
     - `1`
     - :part:`75%`
   * - mlir/lib/Dialect/X86Vector/IR
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/lib/Dialect/X86Vector/Transforms
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/lib/ExecutionEngine
     - `9`
     - `8`
     - `1`
     - :part:`88%`
   * - mlir/lib/Interfaces
     - `11`
     - `11`
     - `0`
     - :good:`100%`
   * - mlir/lib/IR
     - `35`
     - `34`
     - `1`
     - :part:`97%`
   * - mlir/lib/Parser
     - `13`
     - `13`
     - `0`
     - :good:`100%`
   * - mlir/lib/Pass
     - `8`
     - `7`
     - `1`
     - :part:`87%`
   * - mlir/lib/Reducer
     - `4`
     - `4`
     - `0`
     - :good:`100%`
   * - mlir/lib/Rewrite
     - `4`
     - `3`
     - `1`
     - :part:`75%`
   * - mlir/lib/Support
     - `7`
     - `7`
     - `0`
     - :good:`100%`
   * - mlir/lib/TableGen
     - `18`
     - `16`
     - `2`
     - :part:`88%`
   * - mlir/lib/Target/LLVMIR
     - `6`
     - `6`
     - `0`
     - :good:`100%`
   * - mlir/lib/Target/LLVMIR/Dialect/AMX
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/lib/Target/LLVMIR/Dialect/ArmNeon
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/lib/Target/LLVMIR/Dialect/ArmSVE
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/lib/Target/LLVMIR/Dialect/LLVMIR
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/lib/Target/LLVMIR/Dialect/NVVM
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/lib/Target/LLVMIR/Dialect/OpenACC
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/lib/Target/LLVMIR/Dialect/OpenMP
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/lib/Target/LLVMIR/Dialect/ROCDL
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/lib/Target/LLVMIR/Dialect/X86Vector
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/lib/Target/SPIRV
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - mlir/lib/Target/SPIRV/Deserialization
     - `4`
     - `3`
     - `1`
     - :part:`75%`
   * - mlir/lib/Target/SPIRV/Serialization
     - `4`
     - `3`
     - `1`
     - :part:`75%`
   * - mlir/lib/Tools/mlir-lsp-server
     - `5`
     - `5`
     - `0`
     - :good:`100%`
   * - mlir/lib/Tools/mlir-lsp-server/lsp
     - `6`
     - `4`
     - `2`
     - :part:`66%`
   * - mlir/lib/Tools/mlir-reduce
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/lib/Transforms
     - `23`
     - `19`
     - `4`
     - :part:`82%`
   * - mlir/lib/Transforms/Utils
     - `8`
     - `8`
     - `0`
     - :good:`100%`
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
   * - mlir/tools/mlir-linalg-ods-gen
     - `2`
     - `1`
     - `1`
     - :part:`50%`
   * - mlir/tools/mlir-lsp-server
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
     - `1`
     - `1`
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
     - `22`
     - `21`
     - `1`
     - :part:`95%`
   * - mlir/tools/mlir-translate
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/tools/mlir-vulkan-runner
     - `4`
     - `4`
     - `0`
     - :good:`100%`
   * - mlir/unittests/Analysis
     - `3`
     - `3`
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
   * - mlir/unittests/Dialect/Utils
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/unittests/ExecutionEngine
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/unittests/Interfaces
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/unittests/IR
     - `5`
     - `5`
     - `0`
     - :good:`100%`
   * - mlir/unittests/Pass
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - mlir/unittests/Rewrite
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - mlir/unittests/SDBM
     - `1`
     - `0`
     - `1`
     - :none:`0%`
   * - mlir/unittests/Support
     - `5`
     - `4`
     - `1`
     - :part:`80%`
   * - mlir/unittests/TableGen
     - `4`
     - `3`
     - `1`
     - :part:`75%`
   * - openmp/libomptarget/deviceRTLs
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - openmp/libomptarget/deviceRTLs/amdgcn/src
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - openmp/libomptarget/deviceRTLs/common
     - `8`
     - `8`
     - `0`
     - :good:`100%`
   * - openmp/libomptarget/deviceRTLs/common/include/target
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - openmp/libomptarget/deviceRTLs/common/src
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - openmp/libomptarget/deviceRTLs/nvptx/src
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - openmp/libomptarget/include
     - `5`
     - `5`
     - `0`
     - :good:`100%`
   * - openmp/libomptarget/plugins/amdgpu/impl
     - `16`
     - `16`
     - `0`
     - :good:`100%`
   * - openmp/libomptarget/plugins/amdgpu/src
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - openmp/libomptarget/plugins/common/elf_common
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - openmp/libomptarget/plugins/common/MemoryManager
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - openmp/libomptarget/plugins/cuda/dynamic_cuda
     - `2`
     - `2`
     - `0`
     - :good:`100%`
   * - openmp/libomptarget/plugins/cuda/src
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - openmp/libomptarget/plugins/generic-elf-64bit/src
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - openmp/libomptarget/plugins/remote/include
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - openmp/libomptarget/plugins/remote/lib
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - openmp/libomptarget/plugins/remote/server
     - `3`
     - `3`
     - `0`
     - :good:`100%`
   * - openmp/libomptarget/plugins/remote/src
     - `3`
     - `3`
     - `0`
     - :good:`100%`
   * - openmp/libomptarget/plugins/ve/src
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - openmp/libomptarget/src
     - `8`
     - `8`
     - `0`
     - :good:`100%`
   * - openmp/runtime/doc/doxygen
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - openmp/runtime/src
     - `74`
     - `69`
     - `5`
     - :part:`93%`
   * - openmp/runtime/src/thirdparty/ittnotify
     - `6`
     - `5`
     - `1`
     - :part:`83%`
   * - openmp/runtime/src/thirdparty/ittnotify/legacy
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - openmp/tools/archer
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - openmp/tools/archer/tests/ompt
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - openmp/tools/multiplex
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - openmp/tools/multiplex/tests
     - `1`
     - `1`
     - `0`
     - :good:`100%`
   * - openmp/tools/multiplex/tests/custom_data_storage
     - `2`
     - `1`
     - `1`
     - :part:`50%`
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
     - `25`
     - `25`
     - `0`
     - :good:`100%`
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
     - `9`
     - `0`
     - :good:`100%`
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
     - `68`
     - `1`
     - `67`
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
     - `15`
     - `15`
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
     - `12`
     - `10`
     - :part:`54%`
   * - Total
     - :total:`14753`
     - :total:`7515`
     - :total:`7238`
     - :total:`50%`
