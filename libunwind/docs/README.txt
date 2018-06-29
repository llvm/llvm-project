libunwind Documentation
====================

The libunwind documentation is written using the Sphinx documentation generator. It is
currently tested with Sphinx 1.1.3.

To build the documents into html configure libunwind with the following cmake options:

  * -DLLVM_ENABLE_SPHINX=ON
  * -DLIBUNWIND_INCLUDE_DOCS=ON

After configuring libunwind with these options the make rule `docs-libunwind-html`
should be available.

