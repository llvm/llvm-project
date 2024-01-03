# BOLT Address Translation (BAT)
# Purpose
A regular profile collection for BOLT involves collecting samples from
unoptimized binary. BOLT Address Translation allows collecting profile
from BOLT-optimized binary and using it for optimizing the input (pre-BOLT)
binary.

# Overview
BOLT Address Translation is an extra section (`.note.bolt_bat`) inserted by BOLT
into the output binary containing translation tables and split functions linkage
information. This information enables mapping the profile back from optimized
binary onto the original binary.

# Usage
`--enable-bat` flag controls the generation of BAT section. Sampled profile
needs to be passed along with the optimized binary containing BAT section to
`perf2bolt` which reads BAT section and produces fdata profile for the original
binary. Note that YAML profile generation is not supported since BAT doesn't
contain the metadata for input functions.

# Internals
## Section contents
The section is organized as follows:
- Functions table
  - Address translation tables
- Fragment linkage table

## Construction and parsing
BAT section is created from `BoltAddressTranslation` class which captures
address translation information provided by BOLT linker. It is then encoded as a
note section in the output binary.

During profile conversion when BAT-enabled binary is passed to perf2bolt,
`BoltAddressTranslation` class is populated from BAT section. The class is then
queried by `DataAggregator` during sample processing to reconstruct addresses/
offsets in the input binary.

## Encoding format
The encoding is specified in
[BoltAddressTranslation.h](/bolt/include/bolt/Profile/BoltAddressTranslation.h)
and [BoltAddressTranslation.cpp](/bolt/lib/Profile/BoltAddressTranslation.cpp).

### Layout
The general layout is as follows:
```
Functions table header
|------------------|
|  Function entry  |
| |--------------| |
| | OutOff InOff | |
| |--------------| |
~~~~~~~~~~~~~~~~~~~~

Fragment linkage header
|------------------|
| ColdAddr HotAddr |
~~~~~~~~~~~~~~~~~~~~
```

### Functions table
Header:
| Entry  | Width | Description |
| ------ | ----- | ----------- |
| `NumFuncs` | 4B | Number of functions in the functions table |

The header is followed by Functions table with `NumFuncs` entries.
| Entry  | Width | Description |
| ------ | ------| ----------- |
| `Address` | 8B | Function address in the output binary |
| `NumEntries` | 4B | Number of address translation entries for a function |

Function header is followed by `NumEntries` pairs of offsets for current
function.

### Address translation table
| Entry  | Width | Description |
| ------ | ------| ----------- |
| `OutputAddr` | 4B | Function offset in output binary |
| `InputAddr` | 4B | Function offset in input binary with `BRANCHENTRY` top bit |

`BRANCHENTRY` bit denotes whether a given offset pair is a control flow source
(branch or call instruction). If not set, it signifies a control flow target
(basic block offset).

### Fragment linkage table
Following Functions table, fragment linkage table is encoded to link split
cold fragments with main (hot) fragment.
Header:
| Entry  | Width | Description |
| ------ | ------------ | ----------- |
| `NumColdEntries` | 4B | Number of split functions in the functions table |

`NumColdEntries` pairs of addresses follow:
| Entry  | Width | Description |
| ------ | ------| ----------- |
| `ColdAddress` | 8B | Cold fragment address in output binary |
| `HotAddress` | 8B | Hot fragment address in output binary |
