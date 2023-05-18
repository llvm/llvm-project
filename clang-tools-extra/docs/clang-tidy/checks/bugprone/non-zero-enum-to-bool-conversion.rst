.. title:: clang-tidy - bugprone-non-zero-enum-to-bool-conversion

bugprone-non-zero-enum-to-bool-conversion
=========================================

Detect implicit and explicit casts of ``enum`` type into ``bool`` where ``enum``
type doesn't have a zero-value enumerator. If the ``enum`` is used only to hold
values equal to its enumerators, then conversion to ``bool`` will always result
in ``true`` value. This can lead to unnecessary code that reduces readability
and maintainability and can result in bugs.

May produce false positives if the ``enum`` is used to store other values
(used as a bit-mask or zero-initialized on purpose). To deal with them,
``// NOLINT`` or casting first to the underlying type before casting to ``bool``
can be used.

It is important to note that this check will not generate warnings if the
definition of the enumeration type is not available.
Additionally, C++11 enumeration classes are supported by this check.

Overall, this check serves to improve code quality and readability by identifying
and flagging instances where implicit or explicit casts from enumeration types to
boolean could cause potential issues.

Example
-------

.. code-block:: c++

  enum EStatus {
    OK = 1,
    NOT_OK,
    UNKNOWN
  };

  void process(EStatus status) {
    if (!status) {
      // this true-branch won't be executed
      return;
    }
    // proceed with "valid data"
  }

Options
-------

.. option:: EnumIgnoreList

  Option is used to ignore certain enum types when checking for
  implicit/explicit casts to bool. It accepts a semicolon-separated list of
  (fully qualified) enum type names or regular expressions that match the enum
  type names.
  The default value is an empty string, which means no enums will be ignored.
