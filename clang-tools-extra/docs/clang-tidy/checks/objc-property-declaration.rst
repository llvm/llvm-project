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

   Semicolon-separated list of custom acronyms that can be used as a prefix
   or a suffix of property names.

   By default, appends to the list of default acronyms (
   ``IncludeDefaultAcronyms`` set to ``1``).
   If ``IncludeDefaultAcronyms`` is set to ``0``, instead replaces the
   default list of acronyms.

.. option:: IncludeDefaultAcronyms

   Integer value (defaults to ``1``) to control whether the default
   acronyms are included in the list of acronyms.

   If set to ``1``, the value in ``Acronyms`` is appended to the
   default list of acronyms:

   ``ACL;API;ARGB;ASCII;BGRA;CMYK;DNS;FPS;FTP;GIF;GPS;HD;HDR;HTML;HTTP;HTTPS;HUD;ID;JPG;JS;LAN;LZW;MDNS;MIDI;OS;PDF;PIN;PNG;POI;PSTN;PTR;QA;QOS;RGB;RGBA;RGBX;ROM;RPC;RTF;RTL;SDK;SSO;TCP;TIFF;TTS;UI;URI;URL;VC;VOIP;VPN;VR;WAN;XML``.

   If set to ``0``, the value in ``Acronyms`` replaces the default list
   of acronyms.
