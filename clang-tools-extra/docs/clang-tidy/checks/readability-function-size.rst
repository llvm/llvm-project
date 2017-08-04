.. title:: clang-tidy - readability-function-size

readability-function-size
=========================

`google-readability-function-size` redirects here as an alias for this check.

Checks for large functions based on various metrics.

Options
-------

.. option:: LineThreshold

   Flag functions exceeding this number of lines. The default is `-1` (ignore
   the number of lines).

.. option:: StatementThreshold

   Flag functions exceeding this number of statements. This may differ
   significantly from the number of lines for macro-heavy code. The default is
   `800`.

.. option:: BranchThreshold

   Flag functions exceeding this number of control statements. The default is
   `-1` (ignore the number of branches).

.. option:: ParameterThreshold

   Flag functions that exceed a specified number of parameters. The default
   is `-1` (ignore the number of parameters).

.. option:: NestingThreshold

    Flag compound statements which create next nesting level after
    `NestingThreshold`. This may differ significantly from the expected value
    for macro-heavy code. The default is `-1` (ignore the nesting level).
