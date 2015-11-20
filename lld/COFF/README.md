The PE/COFF Linker
==================

This directory contains a linker for Windows operating system.
Because the fundamental design of this port is different from
the other ports of LLD, this port is separated to this directory.

The linker is command-line compatible with MSVC linker and is
generally 2x faster than that. It can be used to link real-world
programs such as LLD itself or Clang, or even web browsers which
are probably the largest open-source programs for Windows.

This document is also applicable to ELF linker because the linker
shares the same design as this COFF linker.

Overall Design
--------------

This is a list of important data types in this linker.

* SymbolBody

  SymbolBody is a class for symbols. They may be created for symbols
  in object files or in archive file headers. The linker may create
  them out of nothing.

  There are mainly three types of SymbolBodies: Defined, Undefined, or
  Lazy. Defined symbols are for all symbols that are considered as
  "resolved", including real defined symbols, COMDAT symbols, common
  symbols, absolute symbols, linker-created symbols, etc. Undefined
  symbols are for undefined symbols, which need to be replaced by
  Defined symbols by the resolver. Lazy symbols represent symbols we
  found in archive file headers -- which can turn into Defined symbols
  if we read archieve members, but we haven't done that yet.

* Symbol

  Symbol is a pointer to a SymbolBody. There's only one Symbol for
  each unique symbol name (this uniqueness is guaranteed by the symbol
  table). Because SymbolBodies are created for each file
  independently, there can be many SymbolBodies for the same
  name. Thus, the relationship between Symbols and SymbolBodies is 1:N.

  The resolver keeps the Symbol's pointer to always point to the "best"
  SymbolBody. Pointer mutation is the resolve operation in this
  linker.

  SymbolBodies have pointers to their Symbols. That means you can
  always find the best SymbolBody from any SymbolBody by following
  pointers twice. This structure makes it very easy to find
  replacements for symbols. For example, if you have an Undefined
  SymbolBody, you can find a Defined SymbolBody for that symbol just
  by going to its Symbol and then to SymbolBody, assuming the resolver
  have successfully resolved all undefined symbols.

* Chunk

  Chunk represents a chunk of data that will occupy space in an
  output. Each regular section becomes a chunk.
  Chunks created for common or BSS symbols are not backed by sections.
  The linker may create chunks out of nothing to append additional
  data to an output.

  Chunks know about their size, how to copy their data to mmap'ed
  outputs, and how to apply relocations to them. Specifically,
  section-based chunks know how to read relocation tables and how to
  apply them.

* SymbolTable

  SymbolTable is basically a hash table from strings to Symbols, with
  a logic to resolve symbol conflicts. It resolves conflicts by symbol
  type. For example, if we add Undefined and Defined symbols, the
  symbol table will keep the latter. If we add Defined and Lazy
  symbols, it will keep the former. If we add Lazy and Undefined, it
  will keep the former, but it will also trigger the Lazy symbol to
  load the archive member to actually resolve the symbol.

* OutputSection

  OutputSection is a container of Chunks. A Chunk belongs to at most
  one OutputSection.

There are mainly three actors in this linker.

* InputFile

  InputFile is a superclass of file readers. We have a different
  subclass for each input file type, such as regular object file,
  archive file, etc. They are responsible for creating and owning
  SymbolBodies and Chunks.

* Writer

  The writer is responsible for writing file headers and Chunks to a
  file. It creates OutputSections, put all Chunks into them, assign
  unique, non-overlapping addresses and file offsets to them, and then
  write them down to a file.

* Driver

  The linking process is drived by the driver. The driver

  - processes command line options,
  - creates a symbol table,
  - creates an InputFile for each input file and put all symbols in it
    into the symbol table,
  - checks if there's no remaining undefined symbols,
  - creates a writer,
  - and passes the symbol table to the writer to write the result to a
    file.

Performance
-----------

It's generally 2x faster than MSVC link.exe. It takes 3.5 seconds to
self-host on my Xeon 2580 machine. MSVC linker takes 7.0 seconds to
link the same executable. The resulting output is 65MB.
The old LLD is buggy that it produces 120MB executable for some reason,
and it takes 30 seconds to do that.

We believe the performance difference comes from simplification and
optimizations we made to the new port. Notable differences are listed
below.

* Reduced number of relocation table reads

  In the old design, relocation tables are read from beginning to
  construct graphs because they consist of graph edges. In the new
  design, they are not read until we actually apply relocations.

  This simplification has two benefits. One is that we don't create
  additional objects for relocations but instead consume relocation
  tables directly. The other is that it reduces number of relocation
  entries we have to read, because we won't read relocations for
  dead-stripped COMDAT sections. Large C++ programs tend to consist of
  lots of COMDAT sections. In the old design, the time to process
  relocation table is linear to size of input. In this new model, it's
  linear to size of output.

* Reduced number of symbol table lookup

  Symbol table lookup can be a heavy operation because number of
  symbols can be very large and each symbol name can be very long
  (think of C++ mangled symbols -- time to compute a hash value for a
  string is linear to the length.)

  We look up the symbol table exactly only once for each symbol in the
  new design. This is I believe the minimum possible number. This is
  achieved by the separation of Symbol and SymbolBody. Once you get a
  pointer to a Symbol by looking up the symbol table, you can always
  get the latest symbol resolution result by just dereferencing a
  pointer. (I'm not sure if the idea is new to the linker. At least,
  all other linkers I've investigated so far seem to look up hash
  tables or sets more than once for each new symbol, but I may be
  wrong.)

* Reduced number of file visits

  The symbol table implements the Windows linker semantics. We treat
  the symbol table as a bucket of all known symbols, including symbols
  in archive file headers. We put all symbols into one bucket as we
  visit new files. That means we visit each file only once.

  This is different from the Unix linker semantics, in which we only
  keep undefined symbols and visit each file one by one until we
  resolve all undefined symbols. In the Unix model, we have to visit
  archive files many times if there are circular dependencies between
  archives.

* Avoiding creating additional objects or copying data

  The data structures described in the previous section are all thin
  wrappers for classes that LLVM libObject provides. We avoid copying
  data from libObject's objects to our objects. We read much less data
  than before. For example, we don't read symbol values until we apply
  relocations because these values are not relevant to symbol
  resolution. Again, COMDAT symbols may be discarded during symbol
  resolution, so reading their attributes too early could result in a
  waste. We use underlying objects directly where doing so makes
  sense.

Parallelism
-----------

The abovementioned data structures are also chosen with
multi-threading in mind. It should relatively be easy to make the
symbol table a concurrent hash map, so that we let multiple workers
work on symbol table concurrently. Symbol resolution in this design is
a single pointer mutation, which allows the resolver work concurrently
in a lock-free manner using atomic pointer compare-and-swap.

It should also be easy to apply relocations and write chunks concurrently.

We created an experimental multi-threaded linker using the Microsoft
ConcRT concurrency library, and it was able to link itself in 0.5
seconds, so we think the design is promising.

Link-Time Optimization
----------------------

LTO is implemented by handling LLVM bitcode files as object files.
The linker resolves symbols in bitcode files normally. If all symbols
are successfully resolved, it then calls an LLVM libLTO function
with all bitcode files to convert them to one big regular COFF file.
Finally, the linker replaces bitcode symbols with COFF symbols,
so that we can link the input files as if they were in the native
format from the beginning.

The details are described in this document.
http://llvm.org/docs/LinkTimeOptimization.html

Glossary
--------

* RVA

  Short for Relative Virtual Address.

  Windows executables or DLLs are not position-independent; they are
  linked against a fixed address called an image base. RVAs are
  offsets from an image base.

  Default image bases are 0x140000000 for executables and 0x18000000
  for DLLs. For example, when we are creating an executable, we assume
  that the executable will be loaded at address 0x140000000 by the
  loader, so we apply relocations accordingly. Result texts and data
  will contain raw absolute addresses.

* VA

  Short for Virtual Address. Equivalent to RVA + image base. It is
  rarely used. We almost always use RVAs instead.

* Base relocations

  Relocation information for the loader. If the loader decides to map
  an executable or a DLL to a different address than their image
  bases, it fixes up binaries using information contained in the base
  relocation table. A base relocation table consists of a list of
  locations containing addresses. The loader adds a difference between
  RVA and actual load address to all locations listed there.

  Note that this run-time relocation mechanism is much simpler than ELF.
  There's no PLT or GOT. Images are relocated as a whole just
  by shifting entire images in memory by some offsets. Although doing
  this breaks text sharing, I think this mechanism is not actually bad
  on today's computers.

* ICF

  Short for Identical COMDAT Folding.

  ICF is an optimization to reduce output size by merging COMDAT sections
  by not only their names but by their contents. If two COMDAT sections
  happen to have the same metadata, actual contents and relocations,
  they are merged by ICF. It is known as an effective technique,
  and it usually reduces C++ program's size by a few percent or more.

  Note that this is not entirely sound optimization. C/C++ require
  different functions have different addresses. If a program depends on
  that property, it would fail at runtime. However, that's not really an
  issue on Windows because MSVC link.exe enabled the optimization by
  default. As long as your program works with the linker's default
  settings, your program should be safe with ICF.
