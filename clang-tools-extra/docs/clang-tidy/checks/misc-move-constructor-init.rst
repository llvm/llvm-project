.. title:: clang-tidy - misc-move-constructor-init

misc-move-constructor-init
==========================


The check flags user-defined move constructors that have a ctor-initializer
initializing a member or base class through a copy constructor instead of a
move constructor.

It also flags constructor arguments that are passed by value, have a non-deleted
move-constructor and are assigned to a class field by copy construction.
