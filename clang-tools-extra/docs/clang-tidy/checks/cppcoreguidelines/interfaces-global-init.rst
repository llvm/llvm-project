.. title:: clang-tidy - cppcoreguidelines-interfaces-global-init

cppcoreguidelines-interfaces-global-init
========================================

This check flags initializers of globals that access extern objects,
and therefore can lead to order-of-initialization problems.

This rule is part of the "Interfaces" profile of the C++ Core Guidelines, see
https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Ri-global-init

Note that currently this does not flag calls to non-constexpr functions, and
therefore globals could still be accessed from functions themselves.

