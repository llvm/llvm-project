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
- Hot functions table
  - Address translation tables
- Cold functions table

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
Hot functions table header
|------------------|
|  Function entry  |
| |--------------| |
| | OutOff InOff | |
| |--------------| |
~~~~~~~~~~~~~~~~~~~~

Cold functions table header
|------------------|
|  Function entry  |
| |--------------| |
| | OutOff InOff | |
| |--------------| |
~~~~~~~~~~~~~~~~~~~~
```

### Functions table
Hot and cold functions tables share the encoding except difference marked below.
Header:
| Entry  | Encoding | Description |
| ------ | ----- | ----------- |
| `NumFuncs` | ULEB128 | Number of functions in the functions table |

The header is followed by Functions table with `NumFuncs` entries.
Output binary addresses are delta encoded, meaning that only the difference with
the previous output address is stored. Addresses implicitly start at zero.
Hot indices are delta encoded, implicitly starting at zero.
| Entry  | Encoding | Description |
| ------ | ------| ----------- |
| `Address` | Delta, ULEB128 | Function address in the output binary |
| `HotIndex` | Delta, ULEB128 | Cold functions only: index of corresponding hot function in hot functions table |
| `NumEntries` | ULEB128 | Number of address translation entries for a function |

Function header is followed by `NumEntries` pairs of offsets for current
function.

### Address translation table
Delta encoding means that only the difference with the previous corresponding
entry is encoded. Offsets implicitly start at zero.
| Entry  | Encoding | Description |
| ------ | ------| ----------- |
| `OutputOffset` | Delta, ULEB128 | Function offset in output binary |
| `InputOffset` | Delta, SLEB128 | Function offset in input binary with `BRANCHENTRY` LSB bit |

`BRANCHENTRY` bit denotes whether a given offset pair is a control flow source
(branch or call instruction). If not set, it signifies a control flow target
(basic block offset).
