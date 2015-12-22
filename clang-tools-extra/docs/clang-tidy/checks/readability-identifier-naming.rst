.. title:: clang-tidy - readability-identifier-naming

readability-identifier-naming
=============================


Checks for identifiers naming style mismatch.

This check will try to enforce coding guidelines on the identifiers naming.
It supports ``lower_case``, ``UPPER_CASE``, ``camelBack`` and ``CamelCase`` casing
and tries to convert from one to another if a mismatch is detected.

It also supports a fixed prefix and suffix that will be prepended or
appended to the identifiers, regardless of the casing.

Many configuration options are available, in order to be able to create
different rules for different kind of identifier. In general, the
rules are falling back to a more generic rule if the specific case is not
configured.
