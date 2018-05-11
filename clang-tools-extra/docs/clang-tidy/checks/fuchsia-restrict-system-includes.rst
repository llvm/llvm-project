.. title:: clang-tidy - fuchsia-restrict-system-includes

fuchsia-restrict-system-includes
================================

Checks for allowed system includes and suggests removal of any others.

It is important to note that running this check with fixes may break code, as
the fix removes headers. Fixes are applied to source and header files, but not
to system headers.

For example, given the allowed system includes 'a.h,b*':

.. code-block:: c++

  #include <a.h>
  #include <b.h>
  #include <bar.h>
  #include <c.h>    // Warning, as c.h is not explicitly allowed
  
All system includes can be allowed with '*', and all can be disallowed with an
empty string ('').
  
Options
-------

.. option:: Includes

   A string containing a comma separated glob list of allowed include filenames.
   Similar to the -checks glob list for running clang-tidy itself, the two
   wildcard characters are '*' and '-', to include and exclude globs,
   respectively.The default is '*', which allows all includes.
