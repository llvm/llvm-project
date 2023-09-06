.. _implementation-defined-behavior:

===============================
Implementation-defined behavior
===============================

Contains the implementation details of the implementation-defined behavior in
libc++. Implementation-defined is mandated to be documented by the Standard.

.. note:
   This page is far from complete.


Implementation-defined behavior
===============================

Updating the Time Zone Database
-------------------------------

The Standard allows implementations to automatically update the
*remote time zone database*. Libc++ opts not to do that. Instead calling

 - ``std::chrono::remote_version()`` will update the version information of the
   *remote time zone database*,
 - ``std::chrono::reload_tzdb()``, if needed, will update the entire
   *remote time zone database*.

This offers a way for users to update the *remote time zone database* and
give them full control over the process.

Listed in the index of implementation-defined behavior
======================================================

The order of the entries matches the entries in the
`draft of the Standard <http://eel.is/c++draft/impldefindex>`_.
