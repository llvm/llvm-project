BOLT Documentation
====================

The BOLT documentation is written using the Sphinx documentation generator. It
is currently tested with Sphinx 1.1.3.

To build the documents into html configure BOLT with the following cmake options:

  * -DLLVM_ENABLE_SPHINX=ON
  * -DBOLT_INCLUDE_DOCS=ON

After configuring BOLT with these options the make rule `docs-bolt-html` should
be available.

BOLT webpages
-------------

The BOLT webpages use Sphinx with custom HTML templates and CSS for styling
consistent with other LLVM subprojects.

The templates are in bolt/docs/_templates directory and styling is in the
_static/bolt.css file. When adding new pages to index.md, please use explicit
toctree labels to keep the homepage and sidebar links consistent.

To build and preview the site, from the llvm-project root:

  * PYTHONPATH=utils/docs sphinx-build -b html bolt/docs bolt-docs-html
  * python3 -m http.server --directory bolt-docs-html
