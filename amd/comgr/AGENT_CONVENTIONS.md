# Comgr Project Conventions

Conventions for working in `amd/comgr/`. Apply these to all changes
under this directory.

This file is the single source of truth for general Comgr project
conventions. The agent-tool wrappers (`amd/comgr/CLAUDE.md`,
`.cursor/rules/comgr.mdc`) point here so updates only need to be made
once.

For hotswap-subsystem-specific conventions (patch-pass authoring,
B0/A0 rewrite invariants, hotswap test driver), see
[`src/hotswap/HOTSWAP_CONVENTIONS.md`](src/hotswap/HOTSWAP_CONVENTIONS.md).

## 1. Code reuse — Comgr first, LLVM second, custom never

**Reuse existing Comgr APIs before writing new ones.** Notable hits:

- `parseTargetIdentifier()` (`src/comgr.cpp:231`) — parses an ISA string
  (`amdgcn-amd-amdhsa--gfx1250:sramecc+`) into arch / vendor / OS /
  environ / processor / features. Don't re-implement string parsing
  for ISA names.
- `ensureLLVMInitialized()` (`src/comgr.cpp:275`) — initializes the
  AMDGPU target stack with thread-safety. Don't roll your own
  `std::call_once` over `LLVMInitializeAMDGPU*` calls; reuse this.
- `DisassemblyInfo::create()` (`src/comgr-disassembly.cpp:25`) — sets
  up the full MC stack (Target, MRI, MAI, MCII, STI, MCContext,
  MCDisassembler, MCInstPrinter). If a new feature needs a similar
  bundle, refactor `DisassemblyInfo` to share, don't duplicate.
- Small refactors of existing Comgr APIs are **preferred over parallel
  implementations**. If extracting a shared helper enables your reuse
  story, do that.

**Reuse existing LLVM APIs second**, especially the MC layer:

- `MCCodeEmitter::encodeInstruction` for instruction encoding.
- `MCRegisterInfo::regsOverlap` for register overlap checks.
- `llvm::AMDGPU::*` from `llvm/TargetParser/TargetParser.h` for AMDGPU
  target queries (`parseArchAMDGCN`, `getArchNameAMDGCN`, etc.).
- `llvm::object::ELFFile<>` for ELF parsing — don't hand-roll section
  or symbol iteration.
- `llvm/Support/Compiler.h` macros for portable attributes.

**For upstream LLVM APIs that need small reworks** to be usable here:
add a `TODO` comment in the Comgr code **and** file a GitHub issue to
fix upstream. Do not implement a parallel version inside Comgr.

## 2. Code quality

- Follow LLVM coding guidelines (`BasedOnStyle: LLVM` in `.clang-format`).
  Run `clang-format` on changed files before submitting.
- Apply upstream LLVM code review standards (small focused commits,
  meaningful commit messages, no unrelated changes).
- **Avoid Windows-hostile code**:
  - Use `LLVM_ATTRIBUTE_WEAK` (from `llvm/Support/Compiler.h`), not
    `__attribute__((weak))`. MSVC does not understand the GCC syntax
    and will fail to build.
  - No GCC/Clang-only attributes without an LLVM-portable wrapper.
- All assembly / disassembly goes through the MC layer (e.g.,
  `assembleSingleInst`, `parseAsmToMCInsts`). **No hardcoded
  instruction opcodes or encoded byte sequences** — let the asm parser
  resolve them, and round-trip through `MCCodeEmitter::encodeInstruction`
  for any modification.
- When invoking the asm parser, register the SourceMgr with `MCContext`
  via `MCContext::initInlineSourceManager()` so error diagnostics on
  bad input don't crash with `Either SourceMgr should be available`.
- **Mnemonic identity is asm-level, not tablegen-level.**
  `MCInstrInfo::getName(Opcode)` returns the tablegen pseudo name —
  on gfx1250 the assembled `v_nop` has opcode name `V_NOP_e32_gfx12`,
  not `V_NOP_e32`. Resolve opcodes once at init via the asm parser,
  cache the resolved opcode / `MCInst`, and compare against the
  cached value. Verify any new mnemonic comparison with
  `llvm-mc -show-inst` against the target you care about.

**Style.** Comgr follows the canonical LLVM coding conventions; for
the full reference Read these source files (in this monorepo):

- [`llvm/docs/CodingStandards.rst`](../../llvm/docs/CodingStandards.rst)
- [`llvm/docs/AMDGPU/DeveloperGuideline.rst`](../../llvm/docs/AMDGPU/DeveloperGuideline.rst)

Comgr-specific deviations and items recurring in code review that
aren't covered upstream:

- Avoid `auto`. Comgr leans stricter than upstream LLVM here — spell
  types out, including iterator types.
- ASCII only in source comments — no box-drawing dividers, no smart
  quotes, no em-dashes.
- Pass `MCRegister` (not `unsigned`) until you need the encoded id.

**Error-return policy.** Don't mix signaling styles within one PR.

- Public C API: `amd_comgr_status_t`.
- Internal helpers: `bool` only when there is one meaningful failure
  mode; otherwise `std::optional<T>` or `llvm::Expected<T>`.
- **No silent returns on failure.** Every failure path emits a specific
  `log()` message naming what was attempted and why it failed.
- Don't return a count where 0 conflates "no candidates" with "found
  candidates but couldn't process them" — that distinction matters to
  downstream callers.

## 3. Testing

Comgr has three test suites:

- `test/` — legacy CTest C-based tests.
- `test-lit/` — LIT integration tests with `comgr-sources/` tool binaries.
- `test-unit/` — newer gtest-based unit tests.

**Prefer LIT tests** over gtests where the public API is reachable.
Compile a kernel with `%clang`, pipe through a LIT tool driver, verify
with `%llvm-objdump` / `%llvm-readelf` / `%FileCheck`.

- LIT inputs should be compiled with **`%clang` directly**, not through
  Comgr actions (`AMD_COMGR_ACTION_ASSEMBLE_SOURCE_TO_RELOCATABLE` etc.).
  Going through Comgr actions implicitly tests the Comgr compiler
  pipeline alongside whatever the test is checking, which makes
  failures harder to attribute.
- **Reuse existing tools in `test-lit/comgr-sources/`** rather than
  adding parallel drivers. Extend an existing tool if needed (e.g.,
  adding a new flag). Add a new tool only when an existing one is
  genuinely a bad fit.
- When extending a shared driver, **unknown-flag handling rejects with
  an error**. Silent pass-through turns typoed RUN lines into false
  negatives.
- gtests in `test-unit/` are appropriate for:
  - Pure functions with bit-level edge cases (e.g., encoding limits).
  - Internal helpers not reachable through any public API path.

`make check-comgr` runs all three suites; the per-suite targets
(`make test`, `make test-lit`, `make test-unit`) run them individually.

**Verify under AddressSanitizer before submitting.** Comgr builds with
`-DADDRESS_SANITIZER=On` (see `CMakeLists.txt`). Re-run `make
check-comgr` against the ASAN build to catch use-after-free, leaks,
and other memory bugs that won't surface in a normal release build.

## 4. PR workflow

**One feature per PR.** Split by file-of-truth — one new
`comgr-*-X.cpp` per PR. Refactors of pre-existing Comgr code go in
their own PR; don't bundle a refactor (e.g. of `DisassemblyInfo`) into
a feature PR.

**Tests required.** Each PR is accompanied by tests aiming for 100%
code coverage of the change being added.

**Keep the branch rebased.** Resolve conflicts before requesting
review.

**Deferral protocol.** When a reviewer asks for a structural change
that would require touching code outside the PR scope:

1. Acknowledge the direction.
2. File a tracking issue and link it inline in the comment thread.
3. State the trigger condition for picking it up.

Deferring without a tracker is the failure mode — the concern gets
silently lost.

## 5. Working as an agent

These items address fingerprints that have surfaced in code review.
Self-audit your diff before pushing.

**Cite, don't claim.** Don't make "verified empirically" or "I
confirmed" claims about LLVM internals (parser behavior, codegen
choices, ABI specifics) without a citation — a link to an MC test, a
TableGen `.td`, or actual `llvm-mc` output. If you can't produce one,
downgrade the claim to "would need to verify" and verify before
letting it become an architectural justification.

**Self-audit for over-decomposition.** Grep your diff for one-line
helpers that wrap a single call (a 4-byte `memcpy` wrapped as a named
function; a `printInst` with one caller). If the helper has one caller
and is one operation, inline it.

**Self-audit for defensive checks against impossible conditions.**
"Defensive against tablegen / decoder corruption" is not a justifiable
guard — if the disassembler is producing garbage, the patch can't
recover. `REQUIRES:` LIT gates that can never fail are dead code.
Delete checks that can't fire.

**Self-audit for duplicated `llvm/` content.** Before adding a
constant, table, X-macro, or helper, grep
`llvm/include/llvm/{Object,BinaryFormat,Support,MC}/` and
`llvm/lib/Target/AMDGPU/Utils/` for the same content. If it exists
and is reachable, use it; if it exists but is target-internal, file a
tracking issue and add a `TODO` linking it.
