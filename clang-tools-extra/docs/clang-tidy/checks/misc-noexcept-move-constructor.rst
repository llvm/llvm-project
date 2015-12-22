.. title:: clang-tidy - misc-noexcept-move-constructor

misc-noexcept-move-constructor
==============================


The check flags user-defined move constructors and assignment operators not
marked with ``noexcept`` or marked with ``noexcept(expr)`` where ``expr``
evaluates to ``false`` (but is not a ``false`` literal itself).

Move constructors of all the types used with STL containers, for example,
need to be declared ``noexcept``. Otherwise STL will choose copy constructors
instead. The same is valid for move assignment operations.
