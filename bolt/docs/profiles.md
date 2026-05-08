# BOLT Profile Formats

BOLT accepts profile data in several formats. This document describes each
format, how to generate it, and how BOLT consumes it.

The general recommended workflow is to convert unsymbolized profiles (perf.data
or pre-aggregated) into symbolized (fdata or YAML):

```
$ perf2bolt executable \
# perf.data is consumed directly:
  -p perf.data
# OR pre-aggregated requires `--pa` switch:
  -p preagg --pa
# fdata is the default output format, YAML is optionally emitted using `-w` flag:
  -o perf.fdata [-w perf.yaml]
# the output format for `-o` can be switched with `--profile-format`:
  -o perf.yaml --profile-format=yaml
```

# Unsymbolized profiles
Sample or trace profiles without symbol information accepted by
perf2bolt, to be converted into symbolized profile formats, used by llvm-bolt.

## Linux perf data

### Collection
Example with brstack:
```bash
perf record -j any,u -e cycles:u -o perf.data -- ./binary
```

### Consumption modes

- **Branch samples (default)**: Branch stack samples from capable hardware
  (Intel LBR, AMD LBRv2/BRS, ARM BRBE).
  Used by default with `perf2bolt` and `llvm-bolt -p perf.data`.
- **Basic aggregation (`-ba`)**: Sample-based profile without branch stacks.
  Lower quality but works on hardware/VMs without branch sampling support.
- **Tracing (`--itrace`)**: Synthesizing branch stacks from trace profile (Intel PT, ARM ETM).
Requires a value (e.g. `i10usl`), see
[perf documentation](https://github.com/torvalds/linux/blob/35f5aa9ccc83f4a4171cdb6ba023e514e2b2ecff/tools/perf/Documentation/itrace.txt)
for details.
- **ARM SPE (`--spe`)**: Statistical Profiling Extension on supported ARM
  platforms providing short (1-deep) branch stacks.

### Build-id verification

BOLT verifies that the build-id in `perf.data` matches the input binary.
Use `--ignore-build-id` to skip this check.

## Pre-aggregated format

Pre-aggregated profile for direct consumption by `perf2bolt --pa` or
`llvm-bolt --pa`. Enables external tools to generate BOLT-compatible profiles
without going through `perf.data`.

### Entry types

```
E <event>
S <start> <count>
[TR] <branch> <ft_start> <ft_end> <count>
B <start> <end> <count> <mispred_count>
[Ff] <start> <end> <count>
r <start> <end> <count>
```

Where:
- `E` — Name of the sampling event used for subsequent entries.
- `S` — Aggregated basic sample at `<start>`.
- `T` — Aggregated trace: branch from `<branch>` to `<ft_start>` with a
  fall-through to `<ft_end>`.
- `R` — Aggregated trace originating at a return.
- `B` — Aggregated branch from `<start>` to `<end>`.
- `F` — Aggregated fall-through from `<start>` to `<end>`.
- `f` — Aggregated fall-through with external origin.
- `r` — Aggregated fall-through with external return as origin.

### Trace mapping
Internally, all branch/fall-through formats are represented as traces, with the
following field mapping:
- `B <start> <end>` -> `T <start> <end> BR_ONLY`
- `F <start> <end>` -> `T FT_ONLY <start> <end>`
- `f <start> <end>` -> `T FT_EXTERNAL_ORIGIN <start> <end>`
- `r <start> <end>` -> `R FT_EXTERNAL_RETURN <start> <end>`

The constants have the following values and can be specified directly:
- `BR_ONLY`/`FT_ONLY`: UINT64_MAX: ffffffffffffffff or -1
- `FT_EXTERNAL_ORIGIN`: UINT64_MAX-1: fffffffffffffffe or -2
- `FT_EXTERNAL_RETURN`: UINT64_MAX-2: fffffffffffffffd or -3

### Call continuation fall-throughs
Where applicable, call continuation fall-throughs are extended back to cover
the call site, improving profile continuity:
- `f`: check that the fall-through start is a block start, not an entry point
  or a landing pad.
- `R`/`r`: unconditional.
- `T`: local origin: extend only if it's a return. External origin: use `f`.

Return hint is only needed for external returns (e.g. from a PLT call), where
the address can't be disassembled to distinguish a return from a jump.

### Best practices
Use T/R format. To capture branch type using Linux perf events, set
the `PERF_SAMPLE_BRANCH_TYPE_SAVE` flag in `perf_event_attr::branch_sample_type`.

To represent top of brstack entry with no fall-through, use `BR_ONLY` as `ft_end`.
Such traces can be extended with an average fall-through length by passing
`--impute-trace-fall-through`.

### Location format

Locations have the format `[<buildid>:]<addr>`:
- `<addr>` — Hex vaddr (non-PIE) or offset from the object base load address (PIE).
- `<buildid>:<offset>` — Offset within the object identified by `<buildid>`.
- `X:<addr>` — External address (outside the profiled binary).

Base load address is the address of the first `PT_LOAD` segment in the binary, which
may not be the same as the segment containing code address.

### Examples

Basic samples profile:
```
E cycles
S 41be50 3
E br_inst_retired.near_taken
S 41be60 6
```

Trace profile combining branches and fall-throughs:
```
T 4b196f 4b19e0 4b19ef 2
```

Trace with unknown fall-through:
```
T 4b196f 4b19e0 -1 2
```

Legacy branch profile with separate branches and fall-throughs:
```
F 41be50 41be50 3
F 41be90 41be90 4
B 4b1942 39b57f0 3 0
B 4b196f 4b19e0 2 0
```

### Generation

Pre-aggregated profiles can be generated by external tools. See
[ebpf-bolt](https://github.com/aaupov/ebpf-bolt) for a reference
implementation using eBPF-based collection.

# Symbolized profiles
The profiles accepted by llvm-bolt. fdata is the legacy format, YAML is the rich (metadata-enabled) format.

## fdata format

Plaintext, space-separated branch profile format written by `perf2bolt` and
consumed by `llvm-bolt -data <file>`. Also produced by BOLT instrumentation.

### LBR mode format

Each line records a branch:

```
<is_sym_from> <sym_from> <off_from> <is_sym_to> <sym_to> <off_to> <mispreds> <branches>
```

Where:
- `<is_sym_from>`, `<is_sym_to>`: `1` if the name is an ELF symbol, `0` if
  it is a DSO name. Special values: `2` for local symbols (includes
  filename), `3`/`4`/`5` for memory events.
- `<sym_from>`, `<sym_to>`: Symbol name or DSO name.
- `<off_from>`, `<off_to>`: Hex offset relative to the symbol/DSO.
- `<mispreds>`: Number of branch mispredictions.
- `<branches>`: Total number of branches.

Example:
```
1 main 3fb 0 /lib/ld-2.21.so 12 4 221
```

### No-LBR mode format

Requires `no_lbr` header followed by an optional event name:

```
no_lbr <event_name>
<is_sym> <sym> <off> <count>
```

### Special headers

- `boltedcollection`: Indicates profile collected on a BOLTed binary.
  Requires BAT (BOLT Address Translation) tables for remapping.

### Memory events format

Memory event types use `<is_sym>` values 3, 4, 5 to record load address
information alongside the instruction location.

## YAML format

Structured profile format with block-level granularity. More resilient to
binary changes and supports stale profile matching.

### Schema

Defined in `ProfileYAMLMapping.h`:

```yaml
header:
  profile-version: <uint32>
  binary-name: <string>
  binary-build-id: <string>        # optional
  profile-flags: [lbr|sample|memevent]
  profile-origin: <string>         # optional, how profile was obtained
  profile-events: <string>         # optional, event names
  dfs-order: <bool>                # optional, default true
  hash-func: <std-hash|xxh3>      # optional, default std-hash
functions:
  - name: <string>
    fid: <uint32>
    hash: <hex64>
    exec: <uint64>
    nblocks: <uint32>
    blocks:
      - bid: <uint32>
        insns: <uint32>
        hash: <hex64>              # optional
        exec: <uint64>             # optional
        succ: [{bid, cnt, mis}]    # optional
        calls: [{off, fid, cnt}]   # optional
    inline_tree: [...]             # optional, pseudo probe info
```

### Hash functions

- `std-hash`: Standard hash function (default for backward compatibility).
- `xxh3`: XXH3 hash function (recommended, better distribution).

### Stale profile matching

BOLT supports matching profiles to modified binaries using block hashes and
call graph matching. When the binary changes between profile collection and
optimization, BOLT uses the hash values to find corresponding blocks in the
new binary.
