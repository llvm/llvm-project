===================
Protocol extensions
===================

.. contents::

clangd supports some features that are not in the official
`Language Server Protocol specification
<https://microsoft.github.io/language-server-protocol/specification>`__.

We cautious about adding extensions. The most important considerations are:

- **Editor support**: How many users will the feature be available to?
- **Standardization**: Is the feature stable? Is it likely to be adopted by more
  editors over time?
- **Utility**: Does the feature provide a lot of value?
- **Complexity**: Is this hard to implement in clangd, or constrain future work?
  Is the protocol complicated?

These extensions may evolve or disappear over time. If you use them, try to
recover gracefully if the structures aren't what's expected.

Switch between the implementation file and the header
=====================================================

*This extension is supported in clangd 6 and newer.*

Switching between the implementation file and the header is an important
feature for C++.  A language server that understands C++ can do a better job
than the editor.

**New client->server request**: ``textDocument/switchSourceHeader``.

Lets editors switch between the main source file (``*.cpp``) and header (``*.h``).

Parameter: ``TextDocumentIdentifier``: an open file.

Result: ``string``: the URI of the corresponding header (if a source file was
provided) or source file (if a header was provided).

If the corresponding file can't be determined, ``""`` is returned.

File status
===========

*This extension is supported in clangd 8 and newer.*

It is important to provide feedback to the user when the UI is not responsive.

This extension provides information about activity on clangd's per-file worker
thread.  This information can be displayed to users to let them know that the
language server is busy with something.  For example, in clangd, building the
AST blocks many other operations.

**New server->client notification**: ``textDocument/clangd.fileStatus``

Sent when the current activity for a file changes. Replaces previous activity
for that file.

Parameter: ``FileStatus`` object with properties:

- ``uri : string``: the document whose status is being updated.
- ``state : string``: human-readable information about current activity.

**New initialization option**: ``initializationOptions.clangdFileStatus : bool``

Enables receiving ``textDocument/clangd.fileStatus`` notifications.

Compilation commands
====================

*This extension is supported in clangd 8 and newer.*

clangd relies on knowing accurate compilation options to correctly interpret a
file. Typically they are found in a ``compile_commands.json`` file in a
directory that contains the file, or an ancestor directory. The following
extensions allow editors to supply the commands over LSP instead.

**New initialization option**: ``initializationOptions.compilationDatabasePath : string``

Specifies the directory containing the compilation database (e.g.,
``compile_commands.json``). This path will be used for all files, instead of
searching their ancestor directories.

**New initialization option**: ``initializationOptions.fallbackFlags : string[]``

Controls the flags used when no specific compile command is found.  The compile
command will be approximately ``clang $FILE $fallbackFlags`` in this case.

**New configuration setting**: ``settings.compilationDatabaseChanges : {string: CompileCommand}``

Provides compile commands for files. This can also be provided on startup as
``initializationOptions.compilationDatabaseChanges``.

Keys are file paths (Not URIs!)

Values are ``{workingDirectory: string, compilationCommand: string[]}``.

Force diagnostics generation
============================

*This extension is supported in clangd 7 and newer.*

Clangd does not regenerate diagnostics for every version of a file (e.g., after
every keystroke), as that would be too slow. Its heuristics ensure:

- diagnostics do not get too stale,
- if you stop editing, diagnostics will catch up.

This extension allows editors to force diagnostics to be generated or not
generated at a particular revision.

**New property of** ``textDocument/didChange`` **request**: ``wantDiagnostics : bool``

- if true, diagnostics will be produced for exactly this version.
- if false, diagnostics will not be produced for this version, even if there
  are no further edits.
- if unset, diagnostics will be produced for this version or some subsequent
  one in a bounded amount of time.

Diagnostic categories
=====================

*This extension is supported in clangd 8 and newer.*

Clang compiler groups diagnostics into categories (e.g., "Inline Assembly
Issue").  Clangd can emit these categories for interested editors.

**New property of** ``Diagnostic`` **object**: ``category : string``:

A human-readable name for a group of related diagnostics.  Diagnostics with the
same code will always have the same category.

**New client capability**: ``textDocument.publishDiagnostics.categorySupport``:

Requests that clangd send ``Diagnostic.category``.

Inline fixes for diagnostics
============================

*This extension is supported in clangd 8 and newer.*

LSP specifies that code actions for diagnostics (fixes) are retrieved
asynchronously using ``textDocument/codeAction``. clangd always computes fixes
eagerly.  Providing them alongside diagnostics can improve the UX in editors.

**New property of** ``Diagnostic`` **object**: ``codeActions : CodeAction[]``:

All the code actions that address this diagnostic.

**New client capability**: ``textDocument.publishDiagnostics.codeActionsInline : bool``

Requests clangd to send ``Diagnostic.codeActions``.

Symbol info request
===================

*This extension is supported in clangd 8 and newer.*

**New client->server request**: ``textDocument/symbolInfo``:

This request attempts to resolve the symbol under the cursor, without
retrieving further information (like definition location, which may require
consulting an index).  This request was added to support integration with
indexes outside clangd.

Parameter: ``TextDocumentPositionParams``

Response: ``SymbolDetails``, an object with properties:

- ``name : string`` the unqualified name of the symbol
- ``containerName : string`` the enclosing namespace, class etc (without
  trailing ``::``)
- ``usr : string``: the clang-specific "unified symbol resolution" identifier
- ``id : string?``: the clangd-specific opaque symbol ID
