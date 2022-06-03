.. title:: clang-tidy - misc-homoglyph

misc-homoglyph
==============

Warn about confusable identifiers, i.e. identifiers that are visually close to
each other, but use different unicode characters. This detetcs potential attack
as described in `Trojan Source <https://www.trojansource.codes>`_.

Example:

.. code-block:: c++

    int fo;
    int 𝐟o;
