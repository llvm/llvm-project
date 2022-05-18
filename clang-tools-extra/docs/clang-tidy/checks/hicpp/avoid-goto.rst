.. title:: clang-tidy - hicpp-avoid-goto

hicpp-avoid-goto
================

The `hicpp-avoid-goto` check is an alias to
`cppcoreguidelines-avoid-goto <../cppcoreguidelines/avoid-goto.html>`_.
Rule `6.3.1 High Integrity C++ <http://www.codingstandard.com/rule/6-3-1-ensure-that-the-labels-for-a-jump-statement-or-a-switch-condition-appear-later-in-the-same-or-an-enclosing-block/>`_
requires that ``goto`` only skips parts of a block and is not used for other
reasons.

Both coding guidelines implement the same exception to the usage of ``goto``.
