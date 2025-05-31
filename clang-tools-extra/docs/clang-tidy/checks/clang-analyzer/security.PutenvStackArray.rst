.. title:: clang-tidy - clang-analyzer-security.PutenvStackArray
.. meta::
   :http-equiv=refresh: 5;URL=https://clang.llvm.org/docs/analyzer/checkers.html#security-putenvstackarray-c

clang-analyzer-security.PutenvStackArray
========================================

Finds calls to the putenv function which pass a pointer to a stack-allocated
(automatic) array as the argument. Function putenv does not copy the passed
string, only a pointer to the data is stored and this data can be read even by
other threads. Content of a stack-allocated array is likely to be overwritten
after exiting from the function.

The `clang-analyzer-security.PutenvStackArray` check is an alias, please see
`Clang Static Analyzer Available Checkers
<https://clang.llvm.org/docs/analyzer/checkers.html#security-putenvstackarray-c>`_
for more information.
