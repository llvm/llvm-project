# DSMIL–Wycheproof Integration (JRTC1-5450-MILSPEC)

This fork of [`google/wycheproof`](https://github.com/google/wycheproof) is extended to act as a
**cryptographic assurance subsystem** inside the DSMIL stack running on an Intel Core Ultra 7 165H
(“JRTC1-5450-MILSPEC”).

This repository **does not** change Wycheproof’s core mission. Instead, it:

- Adds documentation and schemas so that an AI agent can:
  - Run Wycheproof test campaigns as an embedded DSMIL Device 15 (CRYPTO).
  - Generate and execute extended test vectors from AI (Device 47) and quantum (Device 46).
  - Feed results into DSMIL’s data fabric, security AI (Layer 8), and executive dashboards (Layer 9).
- Defines a **DBE (DSMIL Binary Envelope)** message schema slice for the crypto subsystem.
- Specifies how to build and test with a **custom LLVM toolchain (DSLLVM)** for PQC-aware and
  side-channel-conscious builds.

## Upstream and Custom Toolchains

- Upstream Wycheproof: https://github.com/google/wycheproof
- Custom DSMIL LLVM (DSLLVM): https://github.com/SWORDIntel/DSLLVM

This fork is intended to be:

1. **Drop-in compatible** with upstream Wycheproof tests.
2. **Extended** with DSMIL-specific integration hooks (no hard dependency on DSMIL for basic usage).
3. A **reference implementation** for cryptographic assurance in high-security, AI-augmented systems.

## Key Docs

- `docs/DSMIL-WYCHEPROOF.md` – Full DSMIL–Wycheproof integration specification.
- `docs/AI_IMPLEMENTATION_BRIEF.md` – EXACT instructions for an AI engineer/agent.
- `docs/DSLLVM_INTEGRATION.md` – DSLLVM usage and build profiles.
- `docs/MLOPS_INTEGRATION_NOTES.md` – How Wycheproof becomes a gate in the MLOps pipeline.
- `docs/ROADMAP.md` – Phase A–H roadmap for implementation.
- `schemas/*.schema.yaml` – YAML schemas for results, vectors, and summaries.
- `config/*.yaml` – DSMIL device map and DBE message types for crypto.

## Intended Audience

- Human engineers integrating DSMIL with Wycheproof.
- AI agents (e.g., code assistants, automation agents) given this repo and asked to:
  - Implement the full integration plan.
  - Maintain and extend cryptographic assurance capabilities over time.

See `docs/AI_IMPLEMENTATION_BRIEF.md` and `docs/ROADMAP.md` for the task breakdown.
