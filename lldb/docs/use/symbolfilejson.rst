JSON Symbol File Format
=======================

The JSON symbol file format encodes symbols in a text based, human readable
format. JSON symbol files can be used to symbolicate programs that lack symbol
information, for example because they have been stripped.

Under the hood, the JSON symbol file format is also used by the crashlog
script, specifically to provide symbol information for interactive crashlogs.

Format
------

The symbol file consists of a single JSON object with the following top level
keys:

* ``triple`` (string)
* ``uuid`` (string)
* ``type`` (string, optional)
* ``sections`` (array, optional)
* ``symbols`` (array, optional)

The ``triple``, ``uuid`` and ``type`` form the header and should therefore come
first. The ``type`` field is optional. The body consists ``sections`` and
``symbols``. Both arrays are optional, and can be omitted and are allowed to be
empty.

triple
``````

The triple is a string with the triple of the object file it corresponds to.
The triple follows the same format as used by LLVM:
``<arch><sub>-<vendor>-<sys>-<env>``.

.. code-block:: JSON

  { "triple": "arm64-apple-darwin22.0.0" }

uuid
````

The UUID is a string with the textual representation of the UUID of the object
file it corresponds to. The UUID is represented as outlined in RFC 4122: with
32 hexadecimal digits, displayed in five groups separated by hyphens, in the
form 8-4-4-4-12 for a total of 36 characters (32 alphanumeric characters and
four hyphens).

.. code-block:: JSON

  { "uuid": "2107157B-6D7E-39F6-806D-AECDC15FC533" }

type
````
The optional ``type`` field allows you to specify the type of object file the
JSON file represent. This is often unnecessary, and can be omitted, in which
case the file is considered of the type ``DebugInfo``.

Valid values for the ``type`` field are:

* ``corefile``: A core file that has a checkpoint of a program's execution state.
* ``executable``: A normal executable.
* ``debuginfo``: An object file that contains only debug information.
* ``dynamiclinker``: The platform's dynamic linker executable.
* ``objectfile``: An intermediate object file.
* ``sharedlibrary``: A shared library that can be used during execution.
* ``stublibrary``: A library that can be linked against but not used for execution.
* ``jit``: JIT code that has symbols, sections and possibly debug info.


sections
````````

* ``name``: a string representing the section name.
* ``type``: a string representing the section type (see below).
* ``address``: a number representing the section file address.
* ``size``: a number representing the section size in bytes.

.. code-block:: JSON

  {
      "name": "__TEXT",
      "type": "code",
      "address": 0,
      "size": 546,
  }

The ``type`` field accepts the following values: ``code``, ``container``,
``data``, ``debug``.

symbols
```````

Symbols are JSON objects with the following keys:

* ``name``: a string representing the string name.
* ``value``: a number representing the symbol value.
* ``address``: a number representing the symbol address in a section.
* ``size``: a number representing the symbol size.
* ``type``: an optional string representing the symbol type (see below).

A symbol must contain either a ``value`` or an ``address``. The ``type`` is
optional.

.. code-block:: JSON

  {
      "name": "foo",
      "type": "code",
      "size": 10,
      "address": 4294983544,
  }

The ``type`` field accepts any type in the ``lldb::SymbolType`` enum in
`lldb-enumerations.h <https://lldb.llvm.org/cpp_reference/lldb-enumerations_8h.html>`_
, without the ``eSymbolType``. For example ``code`` maps to ``eSymbolTypeCode``
and ``variableType`` to ``eSymbolTypeVariableType``.

Usage
-----

Symbol files can be added with the ``target symbol add`` command. The triple
and UUID will be used to match it to the correct module.

.. code-block:: shell

  (lldb) target symbol add /path/to/symbol.json
  symbol file '/path/to/symbol.json' has been added to '/path/to/executable'

You can use ``image list`` to confirm that the symbol file has been associated
with the module.

.. code-block:: shell

  (lldb) image list
  [  0] A711AB38-1FB1-38B1-B38B-859352ED2A20 0x0000000100000000 /path/to/executable
        /path/to/symbol.json
  [  1] 4BF76A72-53CC-3E42-8945-4E314C101535 0x00000001800c6000 /usr/lib/dyld


Example
-------

The simplest valid JSON symbol file consists of just a triple and UUID:

.. code-block:: JSON

  {
    "triple": "arm64-apple-macosx15.0.0",
    "uuid": "A711AB38-1FB1-38B1-B38B-859352ED2A20"
  }

A JSON symbol file with symbols for ``main``, ``foo``, and ``bar``.

.. code-block:: JSON

  {
      "triple": "arm64-apple-macosx15.0.0",
      "uuid": "321C6225-2378-3E6D-B6C1-6374DEC6D81A",
      "symbols": [
          {
              "name": "main",
              "type": "code",
              "size": 32,
              "address": 4294983552
          },
          {
              "name": "foo",
              "type": "code",
              "size": 8,
              "address": 4294983544
          },
          {
              "name": "bar",
              "type": "code",
              "size": 0,
              "value": 255
          }
      ]
  }

A symbol file with a symbol ``foo`` belonging to the ``__TEXT`` section.

.. code-block:: JSON

  {
      "triple": "arm64-apple-macosx15.0.0",
      "uuid": "58489DB0-F9FF-4E62-ABD1-A7CCE5DFB879",
      "type": "sharedlibrary",
      "sections": [
          {
              "name": "__TEXT",
              "type": "code",
              "address": 0,
              "size": 546
          }
      ],
      "symbols": [
          {
              "name": "foo",
              "address": 256,
              "size": 17
          }
      ]
  }
