# Adding a New BOLT Backend

A new BOLT backend should follow LLVM's
[policy for adding a new target](https://llvm.org/docs/DeveloperPolicy.html#adding-a-new-target).
New BOLT targets are experimental and are not built by default.

Most target-specific work should be isolated in an `MCPlusBuilder`
implementation under `bolt/lib/Target/<Target>`, together with
platform-specific relocation handling in `bolt/lib/Core/Relocation.cpp`.

Changes to common infrastructure, particularly `MCPlusBuilder`, must be
separate from the target changes and land first. Such changes require thorough
tests and must state their processing-time and peak-memory impact on existing
targets. Measurements should use large binaries, including at least Clang
itself.

A backend does not need to implement every `MCPlusBuilder` interface initially.
However, omitted interfaces must be checked against their callers. Analyses and
passes that depend on missing functionality, such as stack frame analysis or
register liveness, must remain disabled.

BOLT passes are not cleanly partitioned into common and target-specific code.
Enable them for a new target only after auditing their target assumptions and
proving them stable with target-specific and end-to-end tests. Otherwise, keep
them disabled.

RISC targets generally need the `LongJmp` pass for branch relaxation. This
requires correct branch-range checks and short- and long-jump construction,
with tests covering range boundaries, calls, tail calls, and linker relaxation.

The initial backend should include focused tests for instruction analysis,
relocations, CFG construction, rewriting, and each enabled pass. Validation
must include rewriting and executing large real-world binaries, with Clang as
the minimum large-binary test.
