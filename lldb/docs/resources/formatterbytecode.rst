Formatter Bytecode
==================

Background
----------

LLDB provides rich customization options to display data types (see :doc:`/use/variable/`). To use custom data formatters, developers need to edit the global `~/.lldbinit` file to make sure they are found and loaded. In addition to this rather manual workflow, developers or library authors can ship ship data formatters with their code in a format that allows LLDB automatically find them and run them securely.

An end-to-end example of such a workflow is the Swift `DebugDescription` macro (see https://www.swift.org/blog/announcing-swift-6/#debugging ) that translates Swift string interpolation into LLDB summary strings, and puts them into a `.lldbsummaries` section, where LLDB can find them.

This document describes a minimal bytecode tailored to running LLDB formatters. It defines a human-readable assembler representation for the language, an efficient binary encoding, a virtual machine for evaluating it, and format for embedding formatters into binary containers.

Goals
~~~~~

Provide an efficient and secure encoding for data formatters that can be used as a compilation target from user-friendly representations (such as DIL, Swift DebugDescription, or NatVis).

Non-goals
~~~~~~~~~

While humans could write the assembler syntax, making it user-friendly is not a goal. It is meant to be used as a compilation target for higher-level, language-specific affordances.

Design of the virtual machine
-----------------------------

The LLDB formatter virtual machine uses a stack-based bytecode, comparable with DWARF expressions, but with higher-level data types and functions.

The virtual machine has two stacks, a data and a control stack. The control stack is kept separate to make it easier to reason about the security aspects of the virtual machine.

Data types
~~~~~~~~~~

All objects on the data stack must have one of the following data types. These data types are "host" data types, in LLDB parlance.

* *String* (UTF-8)
* *Int* (64 bit)
* *UInt* (64 bit)
* *Object* (Basically an `SBValue`)
* *Type* (Basically an `SBType`)
* *Selector* (One of the predefine functions)

*Object* and *Type* are opaque, they can only be used as a parameters of `call`.

Instruction set
---------------

Stack operations
~~~~~~~~~~~~~~~~

These instructions manipulate the data stack directly.

========  ==========  ===========================
 Opcode    Mnemonic    Stack effect
--------  ----------  ---------------------------
 0x00      `dup`       `(x -> x x)`
 0x01      `drop`      `(x y -> x)`
 0x02      `pick`      `(x ... UInt -> x ... x)`
 0x03      `over`      `(x y -> x y x)`
 0x04      `swap`      `(x y -> y x)`
 0x05      `rot`       `(x y z -> z x y)`
========  ==========  ===========================

Control flow
~~~~~~~~~~~~

These manipulate the control stack and program counter. Both `if` and `ifelse` expect a `UInt` at the top of the data stack to represent the condition.

========  ==========  ============================================================
 Opcode    Mnemonic    Description
--------  ----------  ------------------------------------------------------------
 0x10       `{`        push a code block address onto the control stack
  --        `}`        (technically not an opcode) syntax for end of code block
 0x11      `if`        `(UInt -> )` pop a block from the control stack,
                       if the top of the data stack is nonzero, execute it
 0x12      `ifelse`    `(UInt -> )` pop two blocks from the control stack, if
                       the top of the data stack is nonzero, execute the first,
                       otherwise the second.
 0x13      `return`    pop the entire control stack and return
========  ==========  ============================================================

Literals for basic types
~~~~~~~~~~~~~~~~~~~~~~~~

========  ===========  ============================================================
 Opcode    Mnemonic    Description
--------  -----------  ------------------------------------------------------------
 0x20      `123u`      `( -> UInt)` push an unsigned 64-bit host integer
 0x21      `123`       `( -> Int)` push a signed 64-bit host integer
 0x22      `"abc"`     `( -> String)` push a UTF-8 host string
 0x23      `@strlen`   `( -> Selector)` push one of the predefined function
                       selectors. See `call`.
========  ===========  ============================================================

Conversion operations
~~~~~~~~~~~~~~~~~~~~~

========  ===========  ================================================================
 Opcode    Mnemonic    Description
--------  -----------  ----------------------------------------------------------------
 0x2a      `as_int`   `( UInt -> Int)` reinterpret a UInt as an Int
 0x2b      `as_uint`  `( Int -> UInt)` reinterpret an Int as a UInt
 0x2c      `is_null`  `( Object -> UInt )` check an object for null `(object ? 0 : 1)`
========  ===========  ================================================================


Arithmetic, logic, and comparison operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All of these operations are only defined for `Int` and `UInt` and both operands need to be of the same type. The `>>` operator is an arithmetic shift if the parameters are of type `Int`, otherwise it's a logical shift to the right.

========  ==========  ===========================
 Opcode    Mnemonic    Stack effect
--------  ----------  ---------------------------
 0x30      `+`         `(x y -> [x+y])`
 0x31      `-`          etc ...
 0x32      `*`
 0x33      `/`
 0x34      `%`
 0x35      `<<`
 0x36      `>>`
 0x40      `~`
 0x41      `|`
 0x42      `^`
 0x50      `=`
 0x51      `!=`
 0x52      `<`
 0x53      `>`
 0x54      `=<`
 0x55      `>=`
========  ==========  ===========================

Function calls
~~~~~~~~~~~~~~

For security reasons the list of functions callable with `call` is predefined. The supported functions are either existing methods on `SBValue`, or string formatting operations.

========  ==========  ============================================
 Opcode    Mnemonic    Stack effect
--------  ----------  --------------------------------------------
 0x60      `call`      `(Object argN ... arg0 Selector -> retval)`
========  ==========  ============================================

Method is one of a predefined set of *Selectors*.

====  ============================  ===================================================  ==================================
Sel.  Mnemonic                      Stack Effect                                         Description
----  ----------------------------  ---------------------------------------------------  ----------------------------------
0x00  `summary`                     `(Object @summary -> String)`                        `SBValue::GetSummary`
0x01  `type_summary`                `(Object @type_summary -> String)`                   `SBValue::GetTypeSummary`
0x10  `get_num_children`            `(Object @get_num_children -> UInt)`                 `SBValue::GetNumChildren`
0x11  `get_child_at_index`          `(Object UInt @get_child_at_index -> Object)`        `SBValue::GetChildAtIndex`
0x12  `get_child_with_name`         `(Object String @get_child_with_name -> Object)`     `SBValue::GetChildAtIndex`
0x13  `get_child_index`             `(Object String @get_child_index -> UInt)`           `SBValue::GetChildIndex`
0x15  `get_type`                    `(Object @get_type -> Type)`                         `SBValue::GetType`
0x16  `get_template_argument_type`  `(Object UInt @get_template_argument_type -> Type)`  `SBValue::GetTemplateArgumentType`
0x17  `cast`                        `(Object Type @cast -> Object)`                      `SBValue::Cast`
0x20  `get_value`                   `(Object @get_value -> Object)`                      `SBValue::GetValue`
0x21  `get_value_as_unsigned`       `(Object @get_value_as_unsigned -> UInt)`            `SBValue::GetValueAsUnsigned`
0x22  `get_value_as_signed`         `(Object @get_value_as_signed -> Int)`               `SBValue::GetValueAsSigned`
0x23  `get_value_as_address`        `(Object @get_value_as_address -> UInt)`             `SBValue::GetValueAsAddress`
0x40  `read_memory_byte`            `(UInt @read_memory_byte -> UInt)`                   `Target::ReadMemory`
0x41  `read_memory_uint32`          `(UInt @read_memory_uint32 -> UInt)`                 `Target::ReadMemory`
0x42  `read_memory_int32`           `(UInt @read_memory_int32 -> Int)`                   `Target::ReadMemory`
0x43  `read_memory_uint64`          `(UInt @read_memory_uint64 -> UInt)`                 `Target::ReadMemory`
0x44  `read_memory_int64`           `(UInt @read_memory_int64 -> Int)`                   `Target::ReadMemory`
0x45  `read_memory_address`         `(UInt @read_memory_uint64 -> UInt)`                 `Target::ReadMemory`
0x46  `read_memory`                 `(UInt Type @read_memory -> Object)`                 `Target::ReadMemory`
0x50  `fmt`                         `(String arg0 ... @fmt -> String)`                   `llvm::format`
0x51  `sprintf`                     `(String arg0 ... sprintf -> String)`                `sprintf`
0x52  `strlen`                      `(String strlen -> String)`                          `strlen in bytes`
====  ============================  ===================================================  ==================================

Byte Code
~~~~~~~~~

Most instructions are just a single byte opcode. The only exceptions are the literals:

* *String*: Length in bytes encoded as ULEB128, followed length bytes
* *Int*: LEB128
* *UInt*: ULEB128
* *Selector*: ULEB128

Embedding
~~~~~~~~~

Expression programs are embedded into an `.lldbformatters` section (an evolution of the Swift `.lldbsummaries` section) that is a dictionary of type names/regexes and descriptions. It consists of a list of records. Each record starts with the following header:

* Version number (ULEB128)
* Remaining size of the record (minus the header) (ULEB128)

The version number is increased whenever an incompatible change is made. Adding new opcodes or selectors is not an incompatible change since consumers can unambiguously detect this and report an error.

Space between two records may be padded with NULL bytes.

In version 1, a record consists of a dictionary key, which is a type name or regex.

* Length of the key in bytes (ULEB128)
* The key (UTF-8)

A regex has to start with `^`, which is part of the regular expression.

After this comes a flag bitfield, which is a ULEB-encoded `lldb::TypeOptions` bitfield.

* Flags (ULEB128)


This is followed by one or more dictionary values that immediately follow each other and entirely fill out the record size from the header. Each expression program has the following layout:

* Function signature (1 byte)
* Length of the program (ULEB128)
* The program bytecode

The possible function signatures are:

=========  ====================== ==========================
Signature    Mnemonic             Stack Effect
---------  ---------------------- --------------------------
  0x00     `@summary`             `(Object -> String)`
  0x01     `@init`                `(Object -> Object+)`
  0x02     `@get_num_children`    `(Object+ -> UInt)`
  0x03     `@get_child_index`     `(Object+ String -> UInt)`
  0x04     `@get_child_at_index`  `(Object+ UInt -> Object)`
  0x05     `@get_value`           `(Object+ -> String)`
=========  ====================== ==========================

If not specified, the init function defaults to an empty function that just passes the Object along. Its results may be cached and allow common prep work to be done for an Object that can be reused by subsequent calls to the other methods. This way subsequent calls to `@get_child_at_index` can avoid recomputing shared information, for example.

While it is more efficient to store multiple programs per type key, this is not a requirement. LLDB will merge all entries. If there are conflicts the result is undefined.

Execution model
~~~~~~~~~~~~~~~

Execution begins at the first byte in the program. The program counter of the virtual machine starts at offset 0 of the bytecode and may never move outside the range of the program as defined in the header. The data stack starts with one Object or the result of the `@init` function (`Object+` in the table above).

Error handling
~~~~~~~~~~~~~~

In version 1 errors are unrecoverable, the entire expression will fail if any kind of error is encountered.

