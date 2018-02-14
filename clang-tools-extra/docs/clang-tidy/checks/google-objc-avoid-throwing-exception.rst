.. title:: clang-tidy - google-objc-avoid-throwing-exception

google-objc-avoid-throwing-exception
====================================

Finds uses of throwing exceptions usages in Objective-C files.

For the same reason as the Google C++ style guide, we prefer not throwing 
exceptions from Objective-C code.

The corresponding C++ style guide rule:
https://google.github.io/styleguide/cppguide.html#Exceptions

Instead, prefer passing in ``NSError **`` and return ``BOOL`` to indicate success or failure.

A counterexample:

.. code-block:: objc

  - (void)readFile {
    if ([self isError]) {
      @throw [NSException exceptionWithName:...];
    }
  }

Instead, returning an error via ``NSError **`` is preferred:

.. code-block:: objc

  - (BOOL)readFileWithError:(NSError **)error {
    if ([self isError]) {
      *error = [NSError errorWithDomain:...];
      return NO;
    }
    return YES;
  }

The corresponding style guide rule:
http://google.github.io/styleguide/objcguide.html#avoid-throwing-exceptions
