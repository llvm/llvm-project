.. title:: clang-tidy - readability-function-size

readability-function-size
=========================

`google-readability-function-size` redirects here as an alias for this check.

Checks for large functions based on various metrics.

Options
-------

.. option:: LineThreshold (added in 15.0.0)

   Flag functions exceeding this number of lines. The default is `none` (ignore
   the number of lines).

.. option:: StatementThreshold (added in 15.0.0)

   Flag functions exceeding this number of statements. This may differ
   significantly from the number of lines for macro-heavy code. The default is
   `800`.

.. option:: BranchThreshold (added in 15.0.0)

   Flag functions exceeding this number of control statements. The default is
   `none` (ignore the number of branches).

.. option:: ParameterThreshold (added in 15.0.0)

   Flag functions that exceed a specified number of parameters. The default
   is `none` (ignore the number of parameters).

.. option:: NestingThreshold (added in 15.0.0)

    Flag compound statements which create next nesting level after
    `NestingThreshold`. This may differ significantly from the expected value
    for macro-heavy code. The default is `none` (ignore the nesting level).

.. option:: VariableThreshold (added in 15.0.0)

   Flag functions exceeding this number of variables declared in the body.
   The default is `none` (ignore the number of variables).
   Please note that function parameters and variables declared in lambdas,
   GNU Statement Expressions, and nested class inline functions are not counted.

.. option:: CountMemberInitAsStmt (added in 21.1.0)

   When `true`, count class member initializers in constructors as statements.
   Default is `true`.
