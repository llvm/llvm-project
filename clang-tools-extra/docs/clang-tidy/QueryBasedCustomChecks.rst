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

Goals: easy to write, cross platform, multiple versions supported toolkit for
custom clang-tidy rules.
Non-Goals: complex checks, performance, fix-its, etc.

Configuration
=============

`CustomChecks` is a list of custom checks. Each check must contain
  - Name: check name can be used in `-checks` option.
  - Query: `clang-query` string
  - Diagnostic: list of diagnostics to be reported.
    - BindName: name of the node to be bound in `Query`.
    - Message: message to be reported.
    - Level: severity of the diagnostic, the possible values are `Note`, `Warning`.

`CustomChecks` can be configured by `Checks` option in the configuration file.

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

Matters Need Attention
======================

This feature needs to be explicitly enabled by `--experimental-custom-checks`
because it is currently in the experimental stage. Welcome to submit any
suggestions in the `link <https://discourse.llvm.org/t/support-query-based-clang-tidy-external-check/85331>`_.

During the experimental stage, the required configuration structure of this
feature may be changed in the future. Future changes will be as
forward-compatible as possible, but this is not a guarantee.

In subsequent versions, including non-experimental stage, the query statements
will change at any time. The essence of :program:`clang-query` is to parse the
query string and dynamically generate the corresponding AST matcher.
Therefore, its functionality is entirely dependent on the functions provided by
the AST matcher library.
The ast matcher will change along with the changes in the clang AST.
Please refer to `ast matcher reference <https://clang.llvm.org/docs/LibASTMatchersReference.html>`_.
