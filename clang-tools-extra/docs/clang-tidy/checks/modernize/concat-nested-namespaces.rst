.. title:: clang-tidy - modernize-concat-nested-namespaces

modernize-concat-nested-namespaces
==================================

Checks for use of nested namespaces such as ``namespace a { namespace b { ... } }``
and suggests changing to the more concise syntax introduced in C++17: ``namespace a::b { ... }``.
Inline namespaces are not modified.

For example:

.. code-block:: c++

  namespace n1 {
  namespace n2 {
  void t();
  }
  }

  namespace n3 {
  namespace n4 {
  namespace n5 {
  void t();
  }
  }
  namespace n6 {
  namespace n7 {
  void t();
  }
  }
  }

Will be modified to:

.. code-block:: c++

  namespace n1::n2 {
  void t();
  }

  namespace n3 {
  namespace n4::n5 {
  void t();
  }
  namespace n6::n7 {
  void t();
  }
  }

