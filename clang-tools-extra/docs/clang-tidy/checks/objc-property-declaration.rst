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

This check also accepts special acronyms as prefix. Such prefix will suppress
the check of Lower Camel Case according to the guide:
https://developer.apple.com/library/content/documentation/Cocoa/Conceptual/CodingGuidelines/Articles/NamingBasics.html#//apple_ref/doc/uid/20001281-1002931-BBCFHEAB

For a full list of well-known acronyms:
https://developer.apple.com/library/content/documentation/Cocoa/Conceptual/CodingGuidelines/Articles/APIAbbreviations.html#//apple_ref/doc/uid/20001285-BCIHCGAE

The corresponding style rule: https://developer.apple.com/library/content/documentation/Cocoa/Conceptual/CodingGuidelines/Articles/NamingIvarsAndTypes.html#//apple_ref/doc/uid/20001284-1001757

Options
-------

.. option:: Acronyms

   Semicolon-separated list of acronyms that can be used as prefix
   of property names.

   Defaults to `ASCII;PDF;XML;HTML;URL;RTF;HTTP;TIFF;JPG;PNG;GIF;LZW;ROM;RGB;CMYK;MIDI;FTP`.
