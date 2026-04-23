# Comgr Project Conventions

Conventions for working in `amd/comgr/`. Apply these to all changes
under this directory, including the hotswap subsystem.

This file is the single source of truth for Comgr project conventions.
The agent-tool wrappers (`amd/comgr/CLAUDE.md`,
`.cursor/rules/comgr.mdc`) point here so updates only need to be made
once.

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
- All assembly / disassembly should go through MC layer functions
  (e.g., `assembleSingleInst`, `parseAsmToMCInsts` in
  `src/comgr-hotswap-llvm.cpp`). **No hardcoded instruction opcodes** —
  let the asm parser resolve them.
- When invoking the asm parser, register the SourceMgr with `MCContext`
  via `MCContext::initInlineSourceManager()` so error diagnostics on
  bad input don't crash with `Either SourceMgr should be available`.

## 3. Testing

Comgr has three test suites:
- `test/` — legacy CTest C-based tests.
- `test-lit/` — LIT integration tests with `comgr-sources/` tool binaries.
- `test-unit/` — newer gtest-based unit tests.

**Prefer LIT tests** over gtests where the public API is reachable.
Reference pattern: `test-lit/hotswap-rewrite-e2e.hip` — compile a
kernel with `%clang`, pipe through the LIT tool, verify with
`%llvm-objdump` / `%llvm-readelf` / `%FileCheck`.

- LIT inputs should be compiled with **`%clang` directly**, not through
  Comgr actions (`AMD_COMGR_ACTION_ASSEMBLE_SOURCE_TO_RELOCATABLE` etc.).
  Going through Comgr actions implicitly tests the Comgr compiler
  pipeline alongside whatever the test is checking, which makes
  failures harder to attribute.
- **Reuse existing tools in `test-lit/comgr-sources/`** — `hotswap-rewrite`
  is the canonical hotswap input/output driver. Extend an existing tool
  if needed (e.g., adding a new flag). Add a new tool only when an
  existing one is genuinely a bad fit.
- gtests in `test-unit/` are appropriate for:
  - Pure functions with bit-level edge cases (e.g., encoding limits).
  - Internal helpers not reachable through any public API path.

`make check-comgr` runs all three suites; the per-suite targets
(`make test`, `make test-lit`, `make test-unit`) run them individually.

**Verify under AddressSanitizer before submitting.** Comgr builds with
`-DADDRESS_SANITIZER=On` (see `CMakeLists.txt`). Re-run `make
check-comgr` against the ASAN build to catch use-after-free, leaks,
and other memory bugs that won't surface in a normal release build.
