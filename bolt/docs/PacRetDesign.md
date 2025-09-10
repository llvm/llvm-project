# Optimizing binaries with pac-ret hardening

This is a design document about processing the `DW_CFA_AARCH64_negate_ra_state`
DWARF instruction in BOLT. As it describes internal design decisions, the
intended audience is BOLT developers. The document is an updated version of the
[RFC posted on the LLVM Discourse](https://discourse.llvm.org/t/rfc-bolt-aarch64-handle-opnegaterastate-to-enable-optimizing-binaries-with-pac-ret-hardening/86594).


`DW_CFA_AARCH64_negate_ra_state` is also referred to as  `.cfi_negate_ra_state`
in assembly, or `OpNegateRAState` in BOLT sources. In this document, I will use
**negate-ra-state** as a shorthand.

## Introduction

### Pointer Authentication

Refer to the [pac-ret section of the BOLT-binary-analysis document](BinaryAnalysis.md#pac-ret-analysis).

### DW_CFA_AARCH64_negate_ra_state

The negate-ra-state CFI is a vendor-specific Call Frame Instruction defined in
the [Arm ABI](https://github.com/ARM-software/abi-aa/blob/main/aadwarf64/aadwarf64.rst#id1).

```
The DW_CFA_AARCH64_negate_ra_state operation negates bit[0] of the RA_SIGN_STATE pseudo-register.
```

This bit indicates to the unwinder whether the current return address is signed
or not (hence the name). The unwinder uses this information to authenticate the
pointer, and remove the Pointer Authentication Code (PAC) bits. Incorrect
negate-ra-state placement can lead to the unwinder trying to authenticate an
unsigned pointer (which segfaults), or skipping authenticating a signed pointer,
and trying to access an incorrect location (also leading to a segfault).

Note: not *all* unwinders do this. Some use the `xpac` instruction to strip the
PAC bits without authenticating the pointer. This is an incorrect (incomplete)
implementation, as it allows control-flow modification in the case of unwinding.

There are no DWARF instructions to directly set or clear the RA State. However,
two other CFIs can also affect the RA state:
- `DW_CFA_remember_state`: this CFI stores register rules onto an implicit stack.
- `DW_CFA_restore_state`:  this CFI pops rules from this stack.

Example:

| CFI                            | Effect on RA state             |
| ------------------------------ | ------------------------------ |
| (default)                      | 0                              |
| DW_CFA_AARCH64_negate_ra_state | 0 -> 1                         |
| DW_CFA_remember_state          | 1 pushed to the stack          |
| DW_CFA_AARCH64_negate_ra_state | 1 -> 0                         |
| DW_CFA_restore_state           | 0 -> 1 (popped from the stack) |

The Arm ABI also defines the DW_CFA_AARCH64_negate_ra_state_with_pc CFI, but it
is not widely used, and is [likely to become deprecated](https://github.com/ARM-software/abi-aa/issues/327).

### Where are these CFIs needed?

In all locations, where two consecutive instructions have different RA state,
this needs to be indicated to the unwinder. This happens at pointer signing and
authenticating. The other case where two consecutive instructions have different
RA state, but neither of them is signing or authenticating means that they are
not next to each other in control flow. One is part of an execution path with
signed RA, the other is part of a path with an unsigned RA.

In the example below, the first BasicBlock ends in a conditional branch, and
jumps to two different BasicBlocks, each with their own authentication, and
return. The instructions on the border of the second and third BasicBlock have
different RA states. The `ret` at the end of the second BasicBlock is in unsigned
state. The start of the third BasicBlock is after the `paciasp` in the control
flow, but before the authentication. In this case, a negate-ra-state is needed
at the end of the second BasicBlock.

```
        +----------------+
        |     paciasp    |
        |                |
        |      b.cc      |
        +--------+-------+
                 |
+----------------+
|                |
|       +--------v-------+
|       |                |
|       |    autiasp     |
|       |      ret       |   // RA: unsigned
|       +----------------+
+----------------+
                 |
        +--------v-------+  // RA: signed
        |                |
        |     autiasp    |
        |      ret       |
        +----------------+
```

> [!important]
> The unwinder does not follow the control flow graph. It reads unwind
> information in the layout order.

Because these locations are dependent on how the function layout looks,
negate-ra-state CFIs will become invalid during BasicBlock reordering.

## Solution design

The patch introduces two new passes:
1. `MarkRAStatesPass`: assigns the RA state to each instruction based on the CFIs
    in the input binary
2. `InsertNegateRAStatePass`: reads those assigned instruction RA states after
    optimizations, and emits `DW_CFA_AARCH64_negate_ra_state` CFIs at the correct
    places: wherever there is a state change between two consecutive instructions
    in the layout order.

To track metadata on individual instructions, the `MCAnnotation` class was
extended. These also have helper functions in `MCPlusBuilder`.

### Saving annotations at CFI reading

CFIs are read and added to BinaryFunctions in `CFIReaderWriter::FillCFIInfoFor`.
At this point, we add MCAnnotations about negate-ra-state, remember-state and
restore-state CFIs to the instructions they refer to. This is to not interfere
with the CFI processing that already happens in BOLT (e.g. remember-state and
restore-state CFIs are removed in `normalizeCFIState` for reasons unrelated to PAC).

As we add the MCAnnotations *to instructions*, we have to account for the case
where the function starts with a CFI altering the RA state. If a function starts
with a negate-ra-state CFI for example, we cannot save the annotation on the
first instruction, because that itself should already be signed. This is why all
BinaryFunctions have an `initialRAState` bool. If the `Offset` the CFI refers to
is zero, we don't store an annotation, but set the `initialRAState` in
`FillCFIInfoFor`. This information is then used in `MarkRAStates`.

### Binaries without DWARF info

In some cases, the DWARF tables are stripped from the binary. These programs
usually have some other unwind-mechanism. To account for code that uses Pointer
Authentication, but does not have DWARF CFIs, the passes only run on functions
that had at least one negate-ra-state CFI. This information is saved on the
functions during CFI reading.

This also makes sure that the passes don't run on functions that do not store
the return address to the stack, and don't need Pointer Authentication, saving
on runtime overhead.

In summary:
- pointer auth is not used: no change, the new passes do not run.
- pointer auth is used, but DWARF info is stripped: no change, the new passes
  do not run.
- pointer auth is used, and we have DWARF CFIs: passes run, and rewrite the
  negate-ra-state CFI.

### MarkRAStates Pass

This pass runs before optimizations reorder anything.

It processes MCAnnotations generated during the CFI reading stage to check if
instructions have either of the three CFIs that can modify RA state:
- negate-ra-state
- remember-state
- restore-state

Then it adds new MCAnnotations to each instruction, indicating their RA state.
Those annotations are:
- Signed
- Unsigned

Below is a simple example, that shows the two different type of annotations:
what we have before the pass, and after it.

| Instruction                   | Before          |  After   |
| ----------------------------- | --------------- | -------- |
| paciasp                       | negate-ra-state | unsigned |
| stp	x29, x30, [sp, #-0x10]! |                 | signed   |
| mov	x29, sp                 |                 | signed   |
| ldp	x29, x30, [sp], #0x10   |                 | signed   |
| autiasp                       | negate-ra-state | signed   |
| ret                           |                 | unsigned |

##### Error handling in MarkRAState Pass:

Whenever the MarkRAStates pass finds inconsistencies in the current
BinaryFunction, it ignores it by calling `BF.setIgnored()`. This prevents BOLT
from optimizing that function, but it will still be emitted as part of the
original section (`.bolt.org.text`) in its original form.

The inconsistencies are as follows:
- finding a `pac*` instruction when already in signed state
- finding an `aut*` instruction when already in unsigned state
- finding `pac*` and `aut*` instructions without `.cfi_negate_ra_state`.

Users will be informed about the number of ignored functions in the pass, the
exact functions ignored, and the found inconsistency.

### InsertNegateRAStatePass

This pass runs after the optimizations are done. In essence, it does the _inverse_
of MarkRAState pass:
1. it reads the RA state annotations attached to the instructions, and
2. whenever the state changes, it adds a PseudoInstruction that holds an
   OpNegateRAState CFI.

##### Covering newly generated instructions:

Some BOLT passes can add new Instructions. In InsertNegateRAStatePass, we have
to know what RA state these have.

The current solution has the `inferUnknownStates` function to cover these, using
a fairly simple strategy: unknown states inherit the last known state. Testing so
far has shown that this implementation is sufficient.

### Optimizations requiring special attention

Marking states before optimizations ensure that instructions can be moved around
freely. The only special case is function splitting. When a function is split,
the split part becomes a new function in the emitted binary. For unwinding to
work, it needs to "replay" all CFIs that lead up to the split point. BOLT does
this for other CFIs. As negate-ra-state is not read (only stored as an Annotation),
we have to do this manually in InsertNegateRAStatePass. Here, if the split part
starts with an instruction that has Signed RA state, we add a negate-ra-state CFI
to indicate this.

## Option to disallow the feature

To aid debugging, we added the `--disallow-pacret` flag. If the flag is used,
and a function `containedNegateRAState()` after `FillCFIInfoFor()`, BOLT exits
with an error. With this flag, the feature is on by default.
