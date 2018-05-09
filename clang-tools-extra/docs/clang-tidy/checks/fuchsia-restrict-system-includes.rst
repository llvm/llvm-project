.. title:: clang-tidy - fuchsia-restrict-system-includes

fuchsia-restrict-system-includes
================================

Checks for allowed system includes and suggests removal of any others. If no
includes are specified, the check will exit without issuing any warnings. 

It is important to note that running this check with fixes may break code, as
the fix removes headers. Fixes are applied to source and header files, but not
to system headers.

Note that the separator for identifying allowed includes is a semi-colon, and
therefore this check is unable to allow an include with a semi-colon in the
filename (e.g. 'foo;bar.h' will be parsed as allowing 'foo' and 'bar.h', and not
as allowing a file called 'foo;bar.h').

For example, given the allowed system includes 'a.h; b.h':

.. code-block:: c++

  #include <a.h>
  #include <b.h>
  #include <c.h>    // Warning, as c.h is not explicitly allowed
  
Options
-------

.. option:: Includes

   A string containing a semi-colon separated list of allowed include filenames.
   The default is an empty string, which allows all includes.
