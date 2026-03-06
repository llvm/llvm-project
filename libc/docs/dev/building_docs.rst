.. _building_docs:

==========================
Building the Documentation
==========================

This page explains how to build the LLVM-libc HTML documentation locally so
you can preview changes before submitting a patch.

Prerequisites
=============

The LLVM documentation build uses `Sphinx <https://www.sphinx-doc.org/>`__.
The key packages required are:

* ``sphinx`` — the documentation generator
* ``furo`` — the theme used by LLVM-libc
* ``myst-parser`` — Markdown support alongside RST
* ``sphinx-reredirects`` — handles page redirect entries in ``conf.py``

**On Debian/Ubuntu**, all required packages are available via apt:

.. code-block:: bash

   sudo apt-get install python3-sphinx python3-myst-parser \
     python3-sphinx-reredirects furo

**On other systems**, install everything from the shared requirements file:

.. code-block:: bash

   pip install -r llvm/docs/requirements.txt

CMake Configuration
===================

Enable the Sphinx documentation build by adding these flags to your CMake
invocation:

.. code-block:: bash

   cmake ../runtimes \
     -DLLVM_ENABLE_PROJECTS="libc" \
     -DLLVM_ENABLE_SPHINX=ON \
     -DLIBC_INCLUDE_DOCS=ON \
     ...

The ``LLVM_ENABLE_SPHINX=ON`` flag enables Sphinx globally for all LLVM
subprojects.  ``LIBC_INCLUDE_DOCS=ON`` is specific to libc and tells CMake to
register the libc doc targets.

Building
========

Once configured, build the HTML docs with:

.. code-block:: bash

   ninja docs-libc-html

The output is written to ``<build-dir>/tools/libc/docs/html/``.  Open
``index.html`` in a browser to view the site.

Header Status Pages (Auto-generated)
=====================================

The per-header implementation status pages under ``docs/headers/`` are
**not** hand-written RST.  They are generated at build time by
``libc/utils/docgen/docgen.py``, which:

1. Reads YAML function definitions from ``libc/src/<header>/*.yaml``.
2. Scans ``libc/src/<header>/`` for ``.cpp`` implementation files.
3. Checks ``libc/include/llvm-libc-macros/`` for macro ``#define`` entries.
4. Emits an RST ``list-table`` showing each symbol's implementation status,
   C standard section, and POSIX link.

If you add a new function and regenerate, these pages update automatically.
Do **not** hand-edit the generated RST files in ``docs/headers/`` — your
changes will be overwritten the next time the docs are built.

Troubleshooting
===============

``Extension error: Could not import extension myst_parser``
   On Debian/Ubuntu: ``sudo apt-get install python3-myst-parser``.
   Otherwise: ``pip install -r llvm/docs/requirements.txt``.

``WARNING: document isn't included in any toctree``
   A new RST/Markdown file needs a ``toctree`` entry.  Add it to the
   appropriate ``index.rst`` or its parent toctree.

``Extension error: No module named 'sphinx_reredirects'``
   Same fix: ``pip install -r llvm/docs/requirements.txt``.
