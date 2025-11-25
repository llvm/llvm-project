# DSLLVM – DSMIL-Aware LLVM Toolchain

[![Upstream](https://img.shields.io/badge/LLVM-upstream%20aligned-262D3A?logo=llvm&logoColor=white)](https://llvm.org/)
[![DSMIL Stack](https://img.shields.io/badge/DSMIL-multi--layer%20architecture-0B8457.svg)](#what-is-dsmil)
[![Quantum Ready](https://img.shields.io/badge/quantum-Qiskit%20%7C%20hybrid-6C2DC7.svg)](#quantum--ai-integration)
[![PQC Profile](https://img.shields.io/badge/CNSA%202.0-ML--KEM--1024%20%E2%80%A2%20ML--DSA--87%20%E2%80%A2%20SHA--384-E67E22.svg)](#pqc--security-posture)
[![AI-Integrated](https://img.shields.io/badge/AI-instrumented%20toolchain-1F7A8C.svg)](#ai--telemetry-hooks)

---

DSLLVM is a **DSMIL-aware build of LLVM** with a small set of targeted extensions:

- keeps the **standard LLVM/Clang toolchain behaviour**;
- adds **optional hooks** for a multi-layer DSMIL system (devices, clearances, and telemetry);
- exposes **AI and quantum-related metadata** to higher layers without changing normal compiler workflows.

If you already know LLVM, you can treat DSLLVM as “LLVM with an opinionated integration layer” rather than a new compiler.

> **Note**  
> This repository is intentionally vague about downstream systems.  

---

## Highlights

- ✅ **LLVM-first design**  
  - Tracks upstream LLVM closely; core passes and IR semantics are unchanged.  
  - Can be used as a regular `clang`/`lld` toolchain for non-DSMIL builds.

- 🛰️ **DSMIL integration points (optional)**  
  - Lightweight annotations and metadata channels to describe:
    - logical device / layer routing,
    - clearance tags,
    - build-time provenance and audit hints.  
  - All of this is **opt-in** and encoded as normal IR / object metadata.

- 🧠 **AI & telemetry hooks**  
  - Build artefacts can carry compact feature metadata for:
    - performance/size profiles,
    - security posture markers,
    - deployment hints to external AI advisors.  
  - No runtime is mandated; DSLLVM just **emits signals** higher layers may consume.

- ⚛️ **Quantum-aware, not quantum-dependent**  
  - Optional metadata path for handing small optimisation / search problems
    to external **Qiskit-based workflows**.  
  - From the compiler’s point of view, this is just structured metadata attached to IR.

- 🔐 **PQC-aligned security profile**  
  - Compiler options and metadata profiles intended to coexist with
    **CNSA 2.0 style suites** (e.g. ML-KEM-1024, ML-DSA-87, SHA-384) without hard-coding any crypto.  
  - DSLLVM does **not** ship cryptography; it exposes knobs and tags so
    downstream toolchains can enforce their own policies.

---

## Language Mix (Indicative)

This repository is still “normal LLVM under the hood”:

| Language | Approx. share |
|---------:|---------------|
| LLVM IR      | ~41.3% |
| C++          | ~31.2% |
| C            | ~13.1% |
| Assembly     | ~9.9%  |
| MLIR         | ~1.5%  |
| Python       | ~0.8%  |
| Other        | ~2.2%  |

(Actual numbers come from GitHub language stats and may drift over time.)

---

## What DSLLVM Is (and Is Not)

**Is:**

- A **minimally invasive** extension layer on top of LLVM/Clang/LLD.
- A way to **tag and describe** builds for a DSMIL-style multi-layer system.
- A place to keep **AI / quantum / PQC-relevant metadata** close to the code that produced the binaries.

**Is *not*:**

- Not a new IR or language.  
- Not a replacement for upstream security guidance or crypto libraries.  
- Not a mandatory runtime or kernel – it’s “just” the compiler side.

---

## Quantum & AI Integration

DSLLVM does **not** execute quantum workloads itself. Instead, it:

- lets you attach **“quantum candidate”** hints to selected optimisation or search problems;
- keeps those hints in IR / object metadata so an external Qiskit pipeline can pick them up;
- allows AI advisors to see **compiler-level features** (size, structure, call-graphs, annotations) without changing the generated machine code.

These features are entirely optional; standard builds can ignore them.

---

## Building & Using DSLLVM

DSLLVM follows the **standard LLVM build flow**:

1. Configure with CMake (out-of-tree build directory).
2. Build with Ninja or Make.
3. Use `clang`/`clang++`/`lld` as usual.

If you don’t enable any DSMIL/AI options, DSLLVM behaves like a regular LLVM toolchain.

---

## Status

- Core compiler functionality: ✅ usable
- DSMIL / AI / quantum metadata hooks: 🧪 experimental, evolving
- Downstream integrations (DSMIL runtime, advisory layers): out of scope for this repo

For most users, DSLLVM can be dropped in as **“LLVM with extra metadata channels”** and left at that.
