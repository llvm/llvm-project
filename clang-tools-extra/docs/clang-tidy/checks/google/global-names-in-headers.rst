.. title:: clang-tidy - google-global-names-in-headers

google-global-names-in-headers
==============================

Flag global namespace pollution in header files. Right now it only triggers on
``using`` declarations and directives.

The relevant style guide section is
https://google.github.io/styleguide/cppguide.html#Namespaces.
