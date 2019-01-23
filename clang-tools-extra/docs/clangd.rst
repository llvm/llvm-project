============
Clangd
============

.. contents::

.. toctree::
   :maxdepth: 1

:program:`Clangd` is an implementation of the `Language Server Protocol
<https://github.com/Microsoft/language-server-protocol>`_ leveraging Clang.
Clangd's goal is to provide language "smartness" features like code completion,
find references, etc. for clients such as C/C++ Editors.

Using Clangd
==================

:program:`Clangd` is not meant to be used by C/C++ developers directly but
rather from a client implementing the protocol. A client would be typically
implemented in an IDE or an editor.

At the moment, `Visual Studio Code <https://code.visualstudio.com/>`_ is mainly
used in order to test :program:`Clangd` but more clients are likely to make
use of :program:`Clangd` in the future as it matures and becomes a production
quality tool. If you are interested in trying :program:`Clangd` in combination
with Visual Studio Code, you can start by `installing Clangd`_ or
`building Clangd`_, then open Visual Studio Code in the clangd-vscode folder and
launch the extension.

Installing Clangd
==================

Packages are available for debian-based distributions, see the `LLVM packages
page <https://apt.llvm.org/>`_. :program:`Clangd` is included in the
`clang-tools` package.
However, it is a good idea to check your distribution's packaging system first
as it might already be available.

Otherwise, you can install :program:`Clangd` by `building Clangd`_ first.

Building Clangd
==================

You can follow the instructions for `building Clang
<https://clang.llvm.org/get_started.html>`_ but "extra Clang tools" is **not**
optional.

Current Status
==================

Many features could be implemented in :program:`Clangd`.
Here is a list of features that could be useful with the status of whether or
not they are already implemented in :program:`Clangd` and specified in the
Language Server Protocol. Note that for some of the features, it is not clear
whether or not they should be part of the Language Server Protocol, so those
features might be eventually developed outside :program:`Clangd` or as an
extension to the protocol.

+-------------------------------------+------------+----------+
| C/C++ Editor feature                |  LSP       |  Clangd  |
+=====================================+============+==========+
| Formatting                          | Yes        |   Yes    |
+-------------------------------------+------------+----------+
| Completion                          | Yes        |   Yes    |
+-------------------------------------+------------+----------+
| Diagnostics                         | Yes        |   Yes    |
+-------------------------------------+------------+----------+
| Fix-its                             | Yes        |   Yes    |
+-------------------------------------+------------+----------+
| Go to Definition                    | Yes        |   Yes    |
+-------------------------------------+------------+----------+
| Signature Help                      | Yes        |   Yes    |
+-------------------------------------+------------+----------+
| Document Highlights                 | Yes        |   Yes    |
+-------------------------------------+------------+----------+
| Rename                              | Yes        |   Yes    |
+-------------------------------------+------------+----------+
| Source hover                        | Yes        |   Yes    |
+-------------------------------------+------------+----------+
| Find References                     | Yes        |   No     |
+-------------------------------------+------------+----------+
| Code Lens                           | Yes        |   No     |
+-------------------------------------+------------+----------+
| Document Symbols                    | Yes        |   Yes    |
+-------------------------------------+------------+----------+
| Workspace Symbols                   | Yes        |   Yes    |
+-------------------------------------+------------+----------+
| Syntax and Semantic Coloring        | No         |   No     |
+-------------------------------------+------------+----------+
| Code folding                        | No         |   No     |
+-------------------------------------+------------+----------+
| Call hierarchy                      | No         |   No     |
+-------------------------------------+------------+----------+
| Type hierarchy                      | No         |   No     |
+-------------------------------------+------------+----------+
| Organize Includes                   | No         |   No     |
+-------------------------------------+------------+----------+
| Quick Assist                        | No         |   No     |
+-------------------------------------+------------+----------+
| Extract Local Variable              | No         |   No     |
+-------------------------------------+------------+----------+
| Extract Function/Method             | No         |   No     |
+-------------------------------------+------------+----------+
| Hide Method                         | No         |   No     |
+-------------------------------------+------------+----------+
| Implement Method                    | No         |   No     |
+-------------------------------------+------------+----------+
| Gen. Getters/Setters                | No         |   No     |
+-------------------------------------+------------+----------+

Editor Integration
==================

Any full-featured Language Server Protocol Client implementation should work
with :program:`Clangd`. This `list
<https://langserver.org/#implementations-client>`_ contains information about
extensions and plugins that are known to work for different editors.

Vim Integration
---------------

LanguageClient-neovim
~~~~~~~~~~~~~~~~~~~~~

One of the options of using :program:`Clangd` in :program:`vim` (or
:program:`nvim`) is to utilize `LanguageClient-neovim
<https://github.com/autozimu/LanguageClient-neovim>`_ plugin. Please see the
`Clangd Wiki page
<https://github.com/autozimu/LanguageClient-neovim/wiki/Clangd>`_ for
instructions.

VSCode Integration
------------------

:program:`VSCode` provides `vscode-clangd
<https://marketplace.visualstudio.com/items?itemName=llvm-vs-code-extensions.vscode-clangd>`_
which is published in Visual Studio Marketplace and can be installed direcetly
from :program:`VSCode`.

Emacs Integration
-----------------

:program:`Emacs` provides `lsp-mode <github.com/emacs-lsp/lsp-mode>`_ and
`Eglot <https://github.com/joaotavora/eglot>`_ plugins for LSP integration.

Getting Involved
==================

A good place for interested contributors is the `Clangd developer mailing list
<https://lists.llvm.org/mailman/listinfo/clangd-dev>`_. For discussions with the
broader community on topics not only related to Clangd, use
`Clang developer mailing list
<https://lists.llvm.org/mailman/listinfo/cfe-dev>`_.
If you're also interested in contributing patches to :program:`Clangd`, take a
look at the `LLVM Developer Policy
<https://llvm.org/docs/DeveloperPolicy.html>`_ and `Code Reviews
<https://llvm.org/docs/Phabricator.html>`_ page. Contributions of new features
to the `Language Server Protocol
<https://github.com/Microsoft/language-server-protocol>`_ itself would also be
very useful, so that :program:`Clangd` can eventually implement them in a
conforming way.
