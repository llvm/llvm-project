==================
LD64 vs LLD-MACHO
==================

This doc lists all significant deliberate differences in behavior between LD64 and LLD-MachO.

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
