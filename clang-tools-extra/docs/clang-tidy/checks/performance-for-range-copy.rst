.. title:: clang-tidy - performance-for-range-copy

performance-for-range-copy
==========================

Finds C++11 for ranges where the loop variable is copied in each iteration but
it would suffice to obtain it by const reference.

The check is only applied to loop variables of types that are expensive to copy
which means they are not trivially copyable or have a non-trivial copy
constructor or destructor.

To ensure that it is safe to replace the copy with const reference the following
heuristic is employed:

1. The loop variable is const qualified.
2. The loop variable is not const, but only const methods or operators are
   invoked on it, or it is used as const reference or value argument in
   constructors or function calls.
