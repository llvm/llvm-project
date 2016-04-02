.. title:: clang-tidy - readability-function-size

readability-function-size
=========================

`google-readability-function-size` redirects here as an alias for this check.

Checks for large functions based on various metrics.

These options are supported:

  * :option:`LineThreshold` - flag functions exceeding this number of lines. The
    default is `-1` (ignore the number of lines).
  * :option:`StatementThreshold` - flag functions exceeding this number of
    statements. This may differ significantly from the number of lines for
    macro-heavy code. The default is `800`.
  * :option:`BranchThreshold` - flag functions exceeding this number of control
    statements. The default is `-1` (ignore the number of branches).
