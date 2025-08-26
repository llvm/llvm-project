.. title:: clang-tidy - bugprone-method-hiding

bugprone-method-hiding
=========================

Finds derived class methods that hide a (non-virtual) base class method.

In order to be considered "hiding", methods must have the same signature
(i.e. the same name, same number of parameters, same parameter types, etc).
Only checks public, non-templated methods. 