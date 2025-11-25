# Repository Guidelines – Kernel Tree

This guide applies to all code under `kernel/` (core DSMIL kernel research tree).

## Layout & Modules

- `core/`: primary DSMIL core, IOCTL paths, and device abstractions.
- `security/`, `safety/`, `enhanced/`: hardened and extended variants; prefer adding features here instead of touching legacy core paths.
- `examples/` and `debug/`: reference implementations and debugging helpers only; do not ship these to production.
- `rust/`: Rust experiments or supporting tools; keep clearly separated from C kernel modules.

## Build & Test

- Default build from `kernel/`: `make` (or see `Makefile.simple` and `QUICKSTART.md` for recommended flows).
- For focused builds, use documented targets in `README.md` / `PRODUCTION_IMPLEMENTATION.md`.
- Kernel tests and validation flows are described in `TESTING_GUIDE.md`; follow those sequences before modifying production paths.
- Do not run experimental targets on production systems; use a VM or lab hardware.

## Coding Style & Safety

- C: 4-space indentation, no tabs, `snake_case` for functions/variables, `ALL_CAPS` macros and constants.
- Keep IOCTL and token-handling logic explicit and defensive; always validate user pointers, sizes, and device ranges.
- Never weaken quarantine or safety checks without updating the relevant docs (`SAFETY`, `TESTING_GUIDE.md`, or `TPM_AUTHENTICATION_GUIDE.md`) and adding tests.

## Testing & Review Expectations

- Add or update tests whenever you change IOCTL behavior, token access rules, or security boundaries.
- Prefer minimal, reviewable patches that clearly separate refactors from behavior changes.
- For changes that affect hardware interaction, describe expected device behavior and safe rollback in the review description.

