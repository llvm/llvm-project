# Quantum Integration with Qiskit – Device 46 Specification

**Version**: 2.1  
**Date**: 2025-11-23  
**Device**: 46 (Quantum Integration) – Layer 7 (EXTENDED)  
**Status**: Design Complete – Implementation Ready (Research / Experimental)

---

## Executive Summary

Device 46 in Layer 7 (EXTENDED) provides **quantum-classical hybrid processing** using Qiskit for *classical simulation* of quantum circuits.

We **do not** have physical quantum hardware; instead we use Qiskit’s **Aer** simulators to:

1. Prototype **quantum-inspired optimization** (VQE/QAOA) for hyperparameters, pruning, and scheduling.  
2. Explore **quantum feature maps** and kernels for anomaly detection and classification.  
3. Provide a **sandbox** for future integration with real quantum backends.

This is a **research adjunct**, not a primary accelerator:

- **Memory Budget (Layer 7)**: 2 GiB logical budget from the 40 GiB Layer-7 pool.  
- **Compute**: 2 P-cores (CPU-bound; TOPS irrelevant).  
- **Qubit Sweet Spot**: 8–12 qubits (statevector), up to ~30 with MPS for select circuits.  
- **Workloads**: Small, high-value optimization / search problems where exponential state-space matters, and problem size fits ≤ ~12 qubits.

Device 46 is explicitly **bandwidth-light** and **isolated** from the main NPU/GPU datapath: its primary cost is CPU time and a small slice of memory, not LPDDR bandwidth.

---

## Table of Contents

1. [Quantum Computing Fundamentals](#1-quantum-computing-fundamentals)  
2. [Qiskit & Simulator Architecture](#2-qiskit--simulator-architecture)  
3. [Device 46 Integration](#3-device-46-integration)  
4. [Hybrid Workflows](#4-hybrid-workflows)  
5. [DSMIL-Relevant Use Cases](#5-dsmil-relevant-use-cases)  
6. [Performance & Limits](#6-performance--limits)  
7. [Implementation API](#7-implementation-api)  
8. [Observability, Guardrails & Future](#8-observability-guardrails--future)

---

## 1. Quantum Computing Fundamentals

### 1.1 Why Quantum Here?

We position Device 46 as a **search/optimization side-arm**, not a general compute engine.

Good fits:

- **Exponential search spaces** with small dimensionality (≤ 10–12 binary variables):
  - Hyperparameter search with a few discrete knobs.
  - Combinatorial choices like “place N models on 3 devices”.
- **QUBO / Ising formulations** (Max-Cut, allocations, simple scheduling).
- **Quantum kernels** where **non-classical feature maps** might capture structure that RBF/linear miss.

Bad fits:

- Anything with **> 15–20 qubits**.  
- Tasks with known fast classical algorithms (e.g. standard regression, linear classifiers).  
- Latency-critical paths (Device 46 is for offline / background optimization, not hot path serving).

### 1.2 Qubit Reminder

- Classical bit: `0` or `1`.  
- Qubit: \|ψ⟩ = α\|0⟩ + β\|1⟩, with |α|² + |β|² = 1 (superposition).  
- N classical bits: 1 state at a time.  
- N qubits: 2ⁿ complex amplitudes simultaneously.

Key phenomena:

1. **Superposition** – parallel amplitude encoding.  
2. **Entanglement** – correlated states across qubits.  
3. **Interference** – amplitudes add/cancel to favor good solutions.  
4. **Measurement** – collapse to classical bitstring.

For us: all of this is **numerically simulated** on CPU.

---

## 2. Qiskit & Simulator Architecture

### 2.1 Stack Overview

