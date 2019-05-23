.. title:: clang-tidy - google-objc-avoid-nsobject-new

google-objc-avoid-nsobject-new
==============================

Finds calls to ``+new`` or overrides of it, which are prohibited by the
Google Objective-C style guide.

The Google Objective-C style guide forbids calling ``+new`` or overriding it in
class implementations, preferring ``+alloc`` and ``-init`` methods to
instantiate objects.

An example:

.. code-block:: objc

  NSDate *now = [NSDate new];
  Foo *bar = [Foo new];

Instead, code should use ``+alloc``/``-init`` or class factory methods.

.. code-block:: objc

  NSDate *now = [NSDate date];
  Foo *bar = [[Foo alloc] init];

This check corresponds to the Google Objective-C Style Guide rule
`Do Not Use +new
<https://google.github.io/styleguide/objcguide.html#do-not-use-new>`_.
