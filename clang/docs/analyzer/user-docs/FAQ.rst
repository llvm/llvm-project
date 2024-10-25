FAQ and How to Deal with Common False Positives
===============================================

.. contents::
   :local:

Custom Assertions
-----------------

Q: How do I tell the analyzer that I do not want the bug being reported here since my custom error handler will safely end the execution before the bug is reached?

You can tell the analyzer that this path is unreachable by teaching it about your `custom assertion handlers <annotations.html#custom_assertions>`_. For example, you can modify the code segment as following:

.. code-block:: c

   void customAssert() __attribute__((analyzer_noreturn));
   int foo(int *b) {
     if (!b)
       customAssert();
     return *b;
   }

Null Pointer Dereference
------------------------

Q: The analyzer reports a null dereference, but I know that the pointer is never null. How can I tell the analyzer that a pointer can never be null?

The reason the analyzer often thinks that a pointer can be null is because the preceding code checked compared it against null. If you are absolutely sure that it cannot be null, remove the preceding check and, preferably, add an assertion as well. For example:

.. code-block:: c

   void usePointer(int *b);
   int foo(int *b) {
     usePointer(b);
     return *b;
   }

Dead Store
----------

Q: How do I tell the static analyzer that I don't care about a specific dead store?

When the analyzer sees that a value stored into a variable is never used, it's going to produce a message similar to this one:

.. code-block:: none

   Value stored to 'x' is never read

You can use the ``(void)x;`` idiom to acknowledge that there is a dead store in your code but you do not want it to be reported in the future.

Unused Instance Variable
------------------------

Q: How do I tell the static analyzer that I don't care about a specific unused instance variable in Objective-C?

When the analyzer sees that a value stored into a variable is never used, it is going to produce a message similar to this one:

.. code-block:: none

   Instance variable 'commonName' in class 'HappyBird' is never used by the methods in its @implementation

You can add ``__attribute__((unused))`` to the instance variable declaration to suppress the warning.

Unlocalized String
------------------

Q: How do I tell the static analyzer that I don't care about a specific unlocalized string?

When the analyzer sees that an unlocalized string is passed to a method that will present that string to the user, it is going to produce a message similar to this one:

.. code-block:: none

   User-facing text should use localized string macro

If your project deliberately uses unlocalized user-facing strings (for example, in a debugging UI that is never shown to users), you can suppress the analyzer warnings (and document your intent) with a function that just returns its input but is annotated to return a localized string:

.. code-block:: objc

   __attribute__((annotate("returns_localized_nsstring")))
   static inline NSString *LocalizationNotNeeded(NSString *s) {
     return s;
   }

You can then call this function when creating your debugging UI:

.. code-block:: objc

   [field setStringValue:LocalizationNotNeeded(@"Debug")];

Some projects may also find it useful to use NSLocalizedString but add "DNL" or "Do Not Localize" to the string contents as a convention:

.. code-block:: objc

   UILabel *testLabel = [[UILabel alloc] init];
   NSString *s = NSLocalizedString(@"Hello <Do Not Localize>", @"For debug purposes");
   [testLabel setText:s];

Dealloc in Manual Retain/Release
--------------------------------

Q: How do I tell the analyzer that my instance variable does not need to be released in -dealloc under Manual Retain/Release?

If your class only uses an instance variable for part of its lifetime, it may maintain an invariant guaranteeing that the instance variable is always released before -dealloc. In this case, you can silence a warning about a missing release by either adding ``assert(_ivar == nil)`` or an explicit release ``[_ivar release]`` (which will be a no-op when the variable is nil) in -dealloc.

Deciding Nullability
--------------------

Q: How do I decide whether a method's return type should be _Nullable or _Nonnull?

Depending on the implementation of the method, this puts you in one of five situations:

1. You actually never return nil.
2. You do return nil sometimes, and callers are supposed to handle that. This includes cases where your method is documented to return nil given certain inputs.
3. You return nil based on some external condition (such as an out-of-memory error), but the client can't do anything about it either.
4. You return nil only when the caller passes input documented to be invalid. That means it's the client's fault.
5. You return nil in some totally undocumented case.

In (1) you should annotate the method as returning a ``_Nonnull`` object.

In (2) the method should be marked ``_Nullable``.

In (3) you should probably annotate the method ``_Nonnull``. Why? Because no callers will actually check for nil, given that they can't do anything about the situation and don't know what went wrong. At this point things have gone so poorly that there's basically no way to recover.

The least happy case is (4) because the resulting program will almost certainly either crash or just silently do the wrong thing. If this is a new method or you control the callers, you can use ``NSParameterAssert()`` (or the equivalent) to check the precondition and remove the nil return. But if you don't control the callers and they rely on this behavior, you should return mark the method ``_Nonnull`` and return nil cast to _Nonnull anyway.

If you're in (5), document it, then figure out if you're now in (2), (3), or (4).

Intentional Nullability Violation
---------------------------------

Q: How do I tell the analyzer that I am intentionally violating nullability?

In some cases, it may make sense for methods to intentionally violate nullability. For example, your method may — for reasons of backward compatibility — chose to return nil and log an error message in a method with a non-null return type when the client violated a documented precondition rather than check the precondition with ``NSAssert()``. In these cases, you can suppress the analyzer warning with a cast:

.. code-block:: objc

   return (id _Nonnull)nil;

Note that this cast does not affect code generation.

Ensuring Loop Body Execution
----------------------------

Q: The analyzer assumes that a loop body is never entered. How can I tell it that the loop body will be entered at least once?

In cases where you know that a loop will always be entered at least once, you can use assertions to inform the analyzer. For example:

.. code-block:: c

   int foo(int length) {
     int x = 0;
     assert(length > 0);
     for (int i = 0; i < length; i++)
       x += 1;
     return length/x;
   }

By adding ``assert(length > 0)`` in the beginning of the function, you tell the analyzer that your code is never expecting a zero or a negative value, so it won't need to test the correctness of those paths.

Suppressing Specific Warnings
-----------------------------

Q: How can I suppress a specific analyzer warning?

When you encounter an analyzer bug/false positive, check if it's one of the issues discussed above or if the analyzer `annotations <annotations.html#custom_assertions>`_ can resolve the issue by helping the static analyzer understand the code better. Second, please `report it <filing_bugs.html>`_ to help us improve user experience.

Sometimes there's really no "good" way to eliminate the issue. In such cases you can "silence" it directly by annotating the problematic line of code with the help of Clang attribute 'suppress':

.. code-block:: c

   int foo() {
     int *x = nullptr;
     ...
     [[clang::suppress]] {
       // all warnings in this scope are suppressed
       int y = *x;
     }

     // null pointer dereference warning suppressed on the next line
     [[clang::suppress]]
     return *x
   }

   int bar(bool coin_flip) {
     // suppress all memory leak warnings about this allocation
     [[clang::suppress]]
     int *result = (int *)malloc(sizeof(int));

     if (coin_flip)
       return 0;      // including this leak path

     return *result;  // as well as this leak path
   }

Excluding Code from Analysis
----------------------------

Q: How can I selectively exclude code the analyzer examines?

When the static analyzer is using clang to parse source files, it implicitly defines the preprocessor macro ``__clang_analyzer__``. One can use this macro to selectively exclude code the analyzer examines. Here is an example:

.. code-block:: c

   #ifndef __clang_analyzer__
   // Code not to be analyzed
   #endif

This usage is discouraged because it makes the code dead to the analyzer from now on. Instead, we prefer that users file bugs against the analyzer when it flags false positives.
