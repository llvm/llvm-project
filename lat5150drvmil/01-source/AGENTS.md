# Repository Guidelines

This document guides contributors working in `01-source`. It applies to all subdirectories unless a more specific `AGENTS.md` overrides it.

## Project Structure & Modules

- Core kernel research lives in `kernel/` (see `kernel/README.md` and `QUICKSTART.md`).
- The out-of-tree driver is in `kernel-driver/`.
- Userspace libraries and tools are in the repo root (`Makefile`, `military_device_*`) and `userspace-tools/`.
- Integration and analysis helpers live in `agentsystems-integration/`, `serena-integration/`, `debugging/`, and `scripts/`.
- Tests and example programs are under `tests/`.

## Build, Test, and Development

- Top-level userspace build: `make` (add `DEBUG=1` for debug builds).
- Top-level tests: `make test` (basic) and `make test-full` (comprehensive).
- Kernel module: `make -C kernel` then `sudo make -C kernel-driver` as needed; use `make load-kernel` / `make unload-kernel` for DSMIL module.
- Rust chaos framework: from `security_chaos_framework/`, use `cargo build` and `cargo test`.
- Utilities test suite: from `userspace-tools/`, build, then run `../tests/test-utils.sh`.

## Coding Style & Naming

- C: follow existing style—4-space indentation, no tabs, `snake_case` for functions/variables, `ALL_CAPS` for macros and constants.
- Python: prefer `black`/PEP 8 style (4 spaces, `snake_case`), keep scripts self-contained with clear `main()` entry points.
- Rust: use `rustfmt` defaults and idiomatic `snake_case` / `CamelCase` types.
- Keep file names descriptive and lowercase with dashes or underscores (e.g., `thermal_guardian.py`, `test_kernel_direct.c`).

## Testing Guidelines

- Prefer fast, targeted tests: e.g., `make test` or specific binaries under `tests/`.
- Python tests can usually be run directly, e.g., `python3 tests/test_chunked_ioctl.py`.
- For userspace utilities, use `tests/test-utils.sh` after building `userspace-tools/`.
- Avoid changes that reduce safety checks, assertions, or quarantine lists without clear justification.

## Commit & Pull Request Practices

- Use focused commits with messages like `component: concise change summary` (e.g., `kernel: tighten ioctl bounds checking`).
- Reference relevant docs or guides (e.g., `kernel/TESTING_GUIDE.md`) in the description when behavior changes.
- For PRs, include: purpose, key changes, testing steps/commands run, and any safety or performance implications.

