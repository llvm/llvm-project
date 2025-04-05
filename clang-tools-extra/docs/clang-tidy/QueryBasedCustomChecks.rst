====================================
Query Based Custom Clang-Tidy Checks
====================================

Introduction
============

This page provides examples of how to add query based custom checks for
:program:`clang-tidy`.

Custom checks are based on :program:`clang-query` syntax. Every custom checks
will be registered in `custom` module to avoid name conflict. They can be
enabled or disabled by the checks option like the built-in checks.

Custom checks support inheritance from parent configurations like other
configuration items.

Configuration
=============

`CustomChecks` is a list of custom checks. Each check must contain
  - Name: check name can been used in `-checks` option.
  - Query: query string
  - Diagnostic: list of diagnostics to be reported.
     - BindName: name of the node to be bound in `Query`.
     - Message: message to be reported.
     - Level: severity of the diagnostic, the possible values are `Note`, `Warning`, `Error`.

Example
=======

.. code-block:: yaml

  Checks: -*,custom-call-main-function
  CustomChecks:
    - Name: call-main-function
      Query: |
          match callExpr(
            callee(
              functionDecl(isMain()).bind("fn")
            )
          ).bind("callee")
      Diagnostic:
        - BindName: fn
          Message: main function.
          Level: Note
        - BindName: callee
          Message: call to main function.
          Level: Warning

.. code-block:: c++

  int main(); // note: main function.

  void bar() {
    main(); // warning: call to main function. [custom-call-main-function]
  }
