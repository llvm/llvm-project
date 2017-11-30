.. title:: clang-tidy - objc-avoid-nserror-init

objc-avoid-nserror-init
=======================

This check will find out improper initialization of NSError objects.

According to Apple developer document, we should always use factory method 
``errorWithDomain:code:userInfo:`` to create new NSError objects instead
of ``[NSError alloc] init]``. Otherwise it will lead to a warning message
during runtime.

The corresponding information about NSError creation: https://developer.apple.com/library/content/documentation/Cocoa/Conceptual/ErrorHandlingCocoa/CreateCustomizeNSError/CreateCustomizeNSError.html
