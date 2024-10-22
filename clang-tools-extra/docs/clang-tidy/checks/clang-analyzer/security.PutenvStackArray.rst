.. title:: clang-tidy - clang-analyzer-security.PutenvStackArray

clang-analyzer-security.PutenvStackArray
========================================

Finds calls to the function 'putenv' which pass a pointer to an automatic
(stack-allocated) array as the argument.

The clang-analyzer-security.PutenvStackArray check is an alias of
Clang Static Analyzer security.PutenvStackArray.
