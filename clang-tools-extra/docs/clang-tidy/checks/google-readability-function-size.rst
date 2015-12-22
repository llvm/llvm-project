.. title:: clang-tidy - google-readability-function-size

google-readability-function-size
================================


Checks for large functions based on various metrics.

These options are supported:

  * ``LineThreshold`` - flag functions exceeding this number of lines. The
    default is ``-1`` (ignore the number of lines).
  * ``StatementThreshold`` - flag functions exceeding this number of
    statements. This may differ significantly from the number of lines for
    macro-heavy code. The default is ``800``.
  * ``BranchThreshold`` - flag functions exceeding this number of control
    statements. The default is ``-1`` (ignore the number of branches).
