# A bytecode for (LLDB) data formatters

## Background

LLDB provides very rich customization options to display data types (see https://lldb.llvm.org/use/variable.html ). To use custom data formatters, developers typically need to edit the global `~/.lldbinit` file to make sure they are found and loaded. An example for this workflow is the `llvm/utils/lldbDataFormatters.py` script. Because of the manual configuration that is involved, this workflow doesn't scale very well. What would be nice is if developers or library authors could ship ship data formatters with their code and LLDB automatically finds them.

In Swift we added the `DebugDescription` macro (see https://www.swift.org/blog/announcing-swift-6/#debugging ) that translates Swift string interpolation into LLDB summary strings, and puts them into a `.lldbsummaries` section, where LLDB can find them. This works well for simple summaries, but doesn't scale to synthetic child providers or summaries that need to perform some kind of conditional logic or computation. The logical next step would be to store full Python formatters instead of summary strings, but Python code is larger and more importantly it is potentially dangerous to just load an execute untrusted Python code in LLDB.

This document describes a minimal bytecode tailored to running LLDB formatters. It defines a human-readable assembler representation for the language, an efficient binary encoding, a virtual machine for evaluating it, and format for embedding formatters into binary containers.

### Goals

Provide an efficient and secure encoding for data formatters that can be used as a compilation target from user-friendly representations (such as DIL, Swift DebugDescription, or NatVis).

### Non-goals

While humans could write the assembler syntax, making it user-friendly is not a goal.

## Design of the virtual machine

The LLDB formatter virtual machine uses a stack-based bytecode, comparable with DWARF expressions, but with higher-level data types and functions.

The virtual machine has two stacks, a data and a control stack. The control stack is kept separate to make it easier to reason about the security aspects of the VM.

### Data types
These data types are "host" data types, in LLDB parlance.
- _String_ (UTF-8)
- _Int_ (64 bit)
- _UInt_ (64 bit)
- _Object_ (Basically an `SBValue`)
- _Type_ (Basically an `SBType`)
- _Selector_ (One of the predefine functions)

_Object_ and _Type_ are opaque, they can only be used as a parameters of `call`.

## Instruction set

### Stack operations

These manipulate the data stack directly.

- `dup  (x -> x x)`
- `drop (x y -> x)`
- `pick (x ... UInt -> x ... x)`
- `over (x y -> y)`
- `swap (x y -> y x)`
- `rot (x y z -> z x y)`

### Control flow

- `{` pushes a code block address onto the control stack
- `}` (technically not an opcode) denotes the end of a code block
- `if` pops a block from the control stack, if the top of the data stack is nonzero, executes it
- `ifelse` pops two blocks from the control stack, if the top of the data stack is nonzero, executes the first, otherwise the second.

### Literals for basic types

- `123u ( -> UInt)` an unsigned 64-bit host integer.
- `123 ( -> Int)` a signed 64-bit host integer.
- `"abc" ( -> String)` a UTF-8 host string.
- `@strlen ( -> Selector)` one of the predefined functions supported by the VM.

### Arithmetic, logic, and comparison operations
- `+ (x y -> [x+y])`
- `-` etc ...
- `*`
- `/`
- `%`
- `<<`
- `>>`
- `shra` (arithmetic shift right)
- `~`
- `|`
- `^`
- `=`
- `!=`
- `<`
- `>`
- `=<`
- `>=`

### Function calls

For security reasons the list of functions callable with `call` is predefined. The supported functions are either existing methods on `SBValue`, or string formatting operations.

- `call (Object arg0 ... Selector -> retval)`

Method is one of a predefined set of _Selectors_
- `(Object @summary -> String)`
- `(Object @type_summary -> String)`

- `(Object @get_num_children -> UInt)`
- `(Object UInt @get_child_at_index -> Object)`
- `(Object String @get_child_index -> UInt)`
- `(Object @get_type -> Type)`
- `(Object UInt @get_template_argument_type -> Type)`
- `(Object @get_value -> Object)`
- `(Object @get_value_as_unsigned -> UInt)`
- `(Object @get_value_as_signed -> Int)`
- `(Object @get_value_as_address -> UInt)`
- `(Object Type @cast -> Object)`

- `(UInt @read_memory_byte -> UInt)`
- `(UInt @read_memory_uint32 -> UInt)`
- `(UInt @read_memory_int32 -> Int)`
- `(UInt @read_memory_unsigned -> UInt)`
- `(UInt @read_memory_signed -> Int)`
- `(UInt @read_memory_address -> UInt)`
- `(UInt Type @read_memory -> Object)`
 
- `(String arg0 ... fmt -> String)`
- `(String arg0 ... sprintf -> String)`
- `(String strlen -> String)`

## Byte Code

Most instructions are just a single byte opcode. The only exceptions are the literals:

- String: Length in bytes encoded as ULEB128, followed length bytes
- Int: LEB128
- UInt: ULEB128
- Selector: ULEB128

### Embedding

Expression programs are embedded into an `.lldbformatters` section (an evolution of the Swift `.lldbsummaries` section) that is a dictionary of type names/regexes and descriptions. It consists of a list of records. Each record starts with the following header:

- version number (ULEB128)
- remaining size of the record (minus the header) (ULEB128)

Space between two records may be padded with NULL bytes.

In version 1, a record consists of a dictionary key, which is type name or regex.

- length of the key in bytes (ULEB128)
- the key (UTF-8)

A regex has to start with `^`.

This is followed by one or more dictionary values that immediately follow each other and entirely fill out the record size from the header. Each expression program has the following layout:

- function signature (1 byte)
- length of the program (ULEB128)
- the program bytecode

The possible function signatures are:
- 0: `@summary (Object -> String)`
- 1: `@init (Object -> Object+)`
- 2: `@get_num_children (Object+ -> UInt)`
- 3: `@get_child_index (Object+ String -> UInt)`
- 4: `@get_child_at_index (Object+ UInt -> Object)`
- FIXME: potentially also `get_value`? (https://lldb.llvm.org/use/variable.html#synthetic-children)

If not specified, the init function defaults to an empty function that just passes the Object along. Its results may be cached and allow common prep work to be done for an Object that can be reused by subsequent calls to the other methods. This way subsequent calls to `@get_child_at_index` can avoid recomputing shared information, for example.

While it is more efficient to store multiple programs per type key, this is not a requirement. LLDB will merge all entries. If there are conflicts the result is undefined.

### Execution model

Execution begins at the first byte in the program. The program counter may never move outside the range of the program as defined in the header. The data stack starts with one Object or the result of the `@init` function (`Object+` in the table above).

## Error handling

In version 1 errors are unrecoverable, the entire expression will fail if any kind of error is encountered.

