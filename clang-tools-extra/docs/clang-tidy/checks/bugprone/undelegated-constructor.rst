.. title:: clang-tidy - bugprone-undelegated-constructor

bugprone-undelegated-constructor
================================

Finds creation of temporary objects in constructors that look like a
function call to another constructor of the same class.

The user most likely meant to use a delegating constructor or base class
initializer.
