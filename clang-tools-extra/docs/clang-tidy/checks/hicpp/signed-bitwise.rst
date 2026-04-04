.. title:: clang-tidy - hicpp-signed-bitwise
.. meta::
   :http-equiv=refresh: 5;URL=../bugprone/signed-bitwise.html

hicpp-signed-bitwise
====================

The `hicpp-signed-bitwise` check is an alias, please see
`bugprone-signed-bitwise <../bugprone/signed-bitwise.html>`_ for more
information.

Options
-------

.. option:: IgnorePositiveIntegerLiterals

   If this option is set to `true`, the check will not warn on bitwise operations with positive integer literals, e.g. `~0`, `2 << 1`, etc.
   Default value is `false`.
