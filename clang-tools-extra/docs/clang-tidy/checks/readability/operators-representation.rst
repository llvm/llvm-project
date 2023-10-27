.. title:: clang-tidy - readability-operators-representation

readability-operators-representation
====================================

Enforces consistent token representation for invoked binary, unary and
overloaded operators in C++ code. The check supports both traditional and
alternative representations of operators, such as ``&&`` and ``and``, ``||``
and ``or``, and so on.

In the realm of C++ programming, developers have the option to choose between
two distinct representations for operators: traditional token representation
and alternative token representation. Traditional tokens utilize symbols,
such as ``&&``, ``||``, and ``!``, while alternative tokens employ more
descriptive words like ``and``, ``or``, and ``not``.

In the following mapping table, a comprehensive list of traditional and
alternative tokens, along with their corresponding representations,
is presented:

.. table:: Token Representation Mapping Table
    :widths: auto

    =========== ===========
    Traditional Alternative
    =========== ===========
    ``&&``      ``and``
    ``&=``      ``and_eq``
    ``&``       ``bitand``
    ``|``       ``bitor``
    ``~``       ``compl``
    ``!``       ``not``
    ``!=``      ``not_eq``
    ``||``      ``or``
    ``|=``      ``or_eq``
    ``^``       ``xor``
    ``^=``      ``xor_eq``
    =========== ===========

Example
-------

.. code-block:: c++

    // Traditional Token Representation:

    if (!a||!b)
    {
        // do something
    }

    // Alternative Token Representation:

    if (not a or not b)
    {
        // do something
    }

Options
-------

Due to the distinct benefits and drawbacks of each representation, the default
configuration doesn't enforce either. Explicit configuration is needed.

To configure check to enforce Traditional Token Representation for all
operators set options to `&&;&=;&;|;~;!;!=;||;|=;^;^=`.

To configure check to enforce Alternative Token Representation for all
operators set options to
`and;and_eq;bitand;bitor;compl;not;not_eq;or;or_eq;xor;xor_eq`.

Developers do not need to enforce all operators, and can mix the representations
as desired by specifying a semicolon-separated list of both traditional and
alternative tokens in the configuration, such as `and;||;not`.

.. option:: BinaryOperators

    This option allows you to specify a semicolon-separated list of binary
    operators for which you want to enforce specific token representation.
    The default value is empty string.

.. option:: OverloadedOperators

    This option allows you to specify a semicolon-separated list of overloaded
    operators for which you want to enforce specific token representation.
    The default value is empty string.
