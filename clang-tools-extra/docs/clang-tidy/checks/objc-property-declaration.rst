.. title:: clang-tidy - objc-property-declaration

objc-property-declaration
=========================

Finds property declarations in Objective-C files that do not follow the pattern
of property names in Apple's programming guide. The property name should be
in the format of Lower Camel Case.

For code:

.. code-block:: objc

   @property(nonatomic, assign) int LowerCamelCase;

The fix will be:

.. code-block:: objc

   @property(nonatomic, assign) int lowerCamelCase;

The check will only fix 'CamelCase' to 'camelCase'. In some other cases we will
only provide warning messages since the property name could be complicated.
Users will need to come up with a proper name by their own.

This check also accepts special acronyms as prefixes or suffixes. Such prefixes or suffixes
will suppress the Lower Camel Case check according to the guide:
https://developer.apple.com/library/content/documentation/Cocoa/Conceptual/CodingGuidelines/Articles/NamingBasics.html#//apple_ref/doc/uid/20001281-1002931-BBCFHEAB

For a full list of well-known acronyms:
https://developer.apple.com/library/content/documentation/Cocoa/Conceptual/CodingGuidelines/Articles/APIAbbreviations.html#//apple_ref/doc/uid/20001285-BCIHCGAE

The corresponding style rule: https://developer.apple.com/library/content/documentation/Cocoa/Conceptual/CodingGuidelines/Articles/NamingIvarsAndTypes.html#//apple_ref/doc/uid/20001284-1001757

The check will also accept property declared in category with a prefix of
lowercase letters followed by a '_' to avoid naming conflict. For example:

.. code-block:: objc

   @property(nonatomic, assign) int abc_lowerCamelCase;

The corresponding style rule: https://developer.apple.com/library/content/qa/qa1908/_index.html


Options
-------

.. option:: Acronyms

   This option is deprecated and ignored.

.. option:: IncludeDefaultAcronyms

   This option is deprecated and ignored.
