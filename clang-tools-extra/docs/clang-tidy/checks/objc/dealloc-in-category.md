.. title:: clang-tidy - objc-dealloc-in-category

objc-dealloc-in-category
========================

Finds implementations of ``-dealloc`` in Objective-C categories. The category
implementation will override any ``-dealloc`` in the class implementation,
potentially causing issues.

Classes implement ``-dealloc`` to perform important actions to deallocate
an object. If a category on the class implements ``-dealloc``, it will
override the class's implementation and unexpected deallocation behavior
may occur.
