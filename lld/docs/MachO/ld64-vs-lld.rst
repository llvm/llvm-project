=================
ld64 vs LLD-MachO
=================

This doc lists all significant deliberate differences in behavior between ld64
and LLD-MachO.

String Literal Deduplication
****************************
ld64 always deduplicates string literals. LLD only does it when the ``--icf=``
or the ``--deduplicate-literals`` flag is passed. Omitting deduplication by
default ensures that our link is as fast as possible. However, it may also break
some programs which have (incorrectly) relied on string deduplication always
occurring. In particular, programs which compare string literals via pointer
equality must be fixed to use value equality instead.

``-no_deduplicate`` Flag
************************
- ld64: This turns off ICF (deduplication pass) in the linker.
- LLD: This turns off ICF and string merging in the linker.

String Alignment
****************
LLD is `slightly less conservative about aligning cstrings
<https://reviews.llvm.org/D121342>`_, allowing it to pack them more compactly.
This should not result in any meaningful semantic difference.

ObjC Symbols Treatment
**********************
There are differences in how LLD and ld64 handle ObjC symbols loaded from
archives.

- ld64:
   1. Duplicate ObjC symbols from the same archives will not raise an error.
      ld64 will pick the first one.
   2. Duplicate ObjC symbols from different archives will raise a "duplicate
      symbol" error.
- LLD: Duplicate symbols, regardless of which archives they are from, will
  raise errors.

Aliases
=======
ld64 treats all aliases as strong extern definitions. Having two aliases of the
same name, or an alias plus a regular extern symbol of the same name, both
result in duplicate symbol errors. LLD does not check for duplicate aliases;
instead we perform alias resolution first, and only then do we check for
duplicate symbols. In particular, we will not report a duplicate symbol error if
the aliased symbols turn out to be weak definitions, but ld64 will.
