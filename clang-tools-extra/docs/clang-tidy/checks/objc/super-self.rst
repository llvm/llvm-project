.. title:: clang-tidy - objc-super-self

objc-super-self
===============

Finds invocations of ``-self`` on super instances in initializers of subclasses
of ``NSObject`` and recommends calling a superclass initializer instead.

Invoking ``-self`` on super instances in initializers is a common programmer
error when the programmer's original intent is to call a superclass
initializer. Failing to call a superclass initializer breaks initializer
chaining and can result in invalid object initialization.

