==================
LD64 vs LLD-MACHO
==================

This doc lists all significant deliberate differences in behavior between LD64 and LLD-MachO.

String literal deduplication
****************************
LD64 always deduplicates string literals. LLD only does it when the `--icf=` or
the `--deduplicate-literals` flag is passed. Omitting deduplication by default
ensures that our link is as fast as possible. However, it may also break some
programs which have (incorrectly) relied on string deduplication always
occurring. In particular, programs which compare string literals via pointer
equality must be fixed to use value equality instead.

``-no_deduplicate`` Flag
**********************
- LD64:
   * This turns off ICF (deduplication pass) in the linker.
- LLD
   * This turns off ICF and string merging in the linker.

ObjC symbols treatment
**********************
There are differences in how LLD and LD64 handle ObjC symbols loaded from archives.

- LD64:
   * Duplicate ObjC symbols from the same archives will not raise an error. LD64 will pick the first one.
   * Duplicate ObjC symbols from different archives will raise a "duplicate symbol" error.
- LLD:
   * Duplicate symbols, regardless of which archives they are from, will raise errors.
