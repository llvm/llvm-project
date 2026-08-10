BOLT Documentation
====================

The BOLT documentation is written using the Sphinx documentation generator. It
is currently tested with Sphinx 1.1.3.

To build the documents into html configure BOLT with the following cmake options:

  * -DLLVM_ENABLE_SPHINX=ON
  * -DBOLT_INCLUDE_DOCS=ON

After configuring BOLT with these options the make rule `docs-bolt-html` should
be available.
