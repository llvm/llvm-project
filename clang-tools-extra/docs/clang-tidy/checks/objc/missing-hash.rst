.. title:: clang-tidy - objc-missing-hash

objc-missing-hash
=================

Finds Objective-C implementations that implement ``-isEqual:`` without also
appropriately implementing ``-hash``.

Apple documentation highlights that objects that are equal must have the same
hash value:
https://developer.apple.com/documentation/objectivec/1418956-nsobject/1418795-isequal?language=objc

Note that the check only verifies the presence of ``-hash`` in scenarios where
its omission could result in unexpected behavior. The verification of the
implementation of ``-hash`` is the responsibility of the developer, e.g.,
through the addition of unit tests to verify the implementation.
