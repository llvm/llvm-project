# DSMIL–Wycheproof Roadmap (Phases A–H)

This roadmap summarises the implementation phases for integrating Wycheproof
into the DSMIL stack, from a single-node prototype to a multi-node,
compiler-aware, mission-aware crypto assurance fabric.

---

## Phase A – Minimal Viable Crypto Assurance (Solo Node, Layer 3)

**Goal:** Device 15 runs stock Wycheproof on at least one major crypto library;
results are structured and stored.

- Implement Device 15 campaign CLI + JSON output.
- Store results in Postgres (long-term) and tmpfs SQLite (hot cache).
- Wire basic Grafana dashboards.
- Manual campaign initiation only.

**Exit:** Stock Wycheproof campaigns run end-to-end; results visible and queryable.

---

## Phase B – AI-Enhanced Wycheproof (Add Device 47)

**Goal:** Layer-7 AI (Device 47) clusters failures and proposes new test families.

- Device 47 ingests `CryptoTestResult` batches.
- Identify invariants and failure clusters.
- Emit manually-curated extended test vectors (AI origin) back to Device 15.

**Exit:** Human-validated AI-generated test families integrated into campaigns.

---

## Phase C – Quantum-Assisted Extension (Add Device 46)

**Goal:** Device 46 uses QAOA/QUBO-style searches to generate edge-case vectors.

- Encode edge-case search spaces for at least one primitive (e.g. ECDSA P-256).
- Use Qiskit Aer to produce candidate vectors.
- Run quantum-origin vectors through Device 15 and record behaviors.

**Exit:** At least one previously unseen anomaly/robustness issue identified.

---

## Phase D – PQC & Layer-8 Integration

**Goal:** Extend Wycheproof to PQC schemes and tie anomalies into Layer-8 security AI.

- Implement PQC Wycheproof suites (ML-KEM-1024, ML-DSA-87, etc.).
- Use Layer-8 correlators to map anomalies to actual TLS/VPN/PQC usage.
- Raise alerts when production crypto uses primitives with failed tests.

**Exit:** PQC paths have coverage and are visible in security dashboards.

---

## Phase E – Full Layer-9 Reporting & MLOps Gating

**Goal:** Make Wycheproof part of both **executive reporting** and **deployment gates**.

- Generate `CryptoAssuranceSummary` per library/build.
- Integrate summaries into MLOps pipelines as promotion gates.
- Present crypto posture (green/amber/red) to Devices 59/60 dashboards.

**Exit:** No crypto build reaches production without passing Wycheproof gates.

---

## Phase F – Multi-Node Crypto Assurance Fabric

**Goal:** Upgrade from a single-node sensor to a **distributed assurance fabric**.

- Deploy Device 15 equivalents on multiple DSMIL nodes.
- Collect results centrally and perform cross-node differential analysis.
- Detect hardware- and platform-specific anomalies.

**Exit:** Multi-node differential reports and global crypto posture views available.

---

## Phase G – Compiler / Side-Channel Feedback Loop (DSLLVM)

**Goal:** Close the loop between Wycheproof anomalies and DSLLVM compiler behavior.

- Include DSLLVM pass-level metadata in results (enabled passes, profiles).
- Add side-channel instrumentation mode (timing and microarchitectural metrics).
- Device 47 suggests DSLLVM flag/pass adjustments to eliminate anomalies.

**Exit:** At least one documented case of “anomaly → DSLLVM tuning → anomaly removed”.

---

## Phase H – Mission-Aware Risk & Governance

**Goal:** Make crypto posture mission-aware, not just globally green/amber/red.

- Extend summaries with mission-specific risk overlays.
- Define policy rules (e.g. ATOMAL crypto must always be green).
- Track historical trends and use them to drive crypto migrations/refactors.

**Exit:** Mission-aware crypto risk and policy-based gating active in executive workflows.
