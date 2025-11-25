# AI Implementation Brief – DSMIL–Wycheproof Integration

**Audience:** AI agent / AI-assisted engineer  
**Context:** This repository is a fork of `google/wycheproof` with additional DSMIL integration docs.

Your mission is to turn this design into a working, tested integration that matches the
`DSMIL–Wycheproof Integration Specification` in `docs/DSMIL-WYCHEPROOF.md`.

---

## 1. Upstream Context

- Upstream Wycheproof repository: `https://github.com/google/wycheproof`
- This fork must:
  - Keep existing Wycheproof tests fully functional.
  - Add DSMIL-specific functionality behind **opt-in** configuration (no breaking changes by default).

**Toolchain Requirement:**  
Use the custom LLVM/Clang toolchain from `https://github.com/SWORDIntel/DSLLVM` where possible.
Standard LLVM/Clang/GCC must still be supported as a fallback.

---

## 2. High-Level Objectives

Implement the following, in order:

1. **Device 15 Wycheproof Execution Engine**
   - Treat Wycheproof test runners as Device 15 in DSMIL Layer 3.
   - Add a machine-friendly CLI/JSON interface to run **campaigns** over specific libraries and primitives.
   - Emit results conforming to `schemas/crypto_test_result.schema.yaml`.

2. **Extended Vector Support (AI & Quantum)**
   - Add a mechanism to ingest **extended test vectors** (AI/Quantum/manual) from JSON/YAML files.
   - Ensure structures conform to:
     - `schemas/crypto_test_vector_classical.schema.yaml`
     - `schemas/crypto_test_vector_pqc.schema.yaml`

3. **DBE Message Integration (Conceptual Layer)**
   - Implement a lightweight adapter that can:
     - Read `config/dbe_message_types.yaml` and `config/dsmil_device_map_wycheproof.yaml`.
     - Map CLI commands and file I/O to the conceptual DBE message types:
       - `CRYPTO_TEST_REQ`, `CRYPTO_TEST_RESULT`, `CRYPTO_TEST_VECTOR_EXT`, `CRYPTO_ASSURANCE_SUMMARY`.

4. **PQC Wycheproof Extensions**
   - Extend or add test suites for PQC algorithms used in DSMIL:
     - `ML-KEM-1024`, `ML-DSA-87`, and any additional PQC from OQS.
   - Ensure that PQC vectors and results conform to the provided PQC schemas.

5. **MLOps Integration Hooks**
   - Provide a CLI or library API such that an external MLOps pipeline can:
     - Trigger campaigns.
     - Parse structured results.
     - Decide whether to “gate” (approve/block) a crypto build.

---

## 3. Concrete Tasks

### 3.1 Device 15 – CLI & JSON Output

1. Implement a new CLI entrypoint, e.g.:
   - `./gradlew :wycheproof:runDsmilCampaign` (Java) **OR**
   - `python3 -m dsmil_wycheproof.run_campaign` (Python, if you add a Python harness)

2. The CLI must:
   - Accept a **campaign definition file** (YAML/JSON) specifying:
     - `lib_id`, `lib_version`, `primitives`, `test_suites`, `options`.
   - Run corresponding Wycheproof tests.
   - Produce a JSON file of test results conforming to `crypto_test_result.schema.yaml`.

3. The CLI must be able to:
   - Run **stock** Wycheproof vectors only.
   - Run **stock + extended** vectors from `schemas`-compatible input files.

### 3.2 Extended Test Vector Support

1. Implement a loader for extended vectors:
   - For classical primitives: `crypto_test_vector_classical.schema.yaml`.
   - For PQC primitives: `crypto_test_vector_pqc.schema.yaml`.

2. The loader should:
   - Validate extended vectors against their schemas.
   - Map vector fields to the appropriate Wycheproof test harness inputs (keys, nonces, messages, etc.).

3. When running tests, tag each result with:
   - `vector_origin` (stock_wycheproof | ai_extended | quantum_extended | manual | fuzz).
   - `vector_id` (from the extended vector file).

### 3.3 DBE Mapping & Config Awareness

1. Read `config/dsmil_device_map_wycheproof.yaml` to know:
   - Which device IDs and layers this subsystem targets (Device 15, 46, 47, 52, 59).

2. Read `config/dbe_message_types.yaml` to know:
   - The conceptual message types and expected fields.

3. Implement simple adapters (no network required in this fork) that:
   - Export `CRYPTO_TEST_RESULT` and `CRYPTO_ASSURANCE_SUMMARY` in a file/stream format that matches the schemas.
   - These adapters **simulate** DBE messages so a higher-level DSMIL bus can consume them.

### 3.4 PQC Test Suites

1. Add new PQC test definitions and harness code for:
   - `ML-KEM-1024`
   - `ML-DSA-87`
   - Any additional PQC schemes used by DSMIL.

2. Use `PqcWycheproofVector` structures from the PQC schema file:
   - Implement key/ciphertext/signature generation from the vector fields.
   - Ensure **expected behavior** is correctly encoded (success/failure semantics).

### 3.5 MLOps Integration Hooks

1. Provide a lightweight library API, for example:
   - `dsmil_wycheproof.run_campaign(campaign_config) -> CryptoAssuranceSummary`

2. Ensure that the returned structure conforms to `crypto_assurance_summary.schema.yaml`.

3. This API should be:
   - Purely programmatic (no interactive input).
   - Deterministic given the same inputs.

---

## 4. Constraints & Non-Goals

- **Do not** break or change the semantics of existing Wycheproof tests.
- **Do not** hard-depend on DSMIL runtime; this fork should still be usable as “Wycheproof+”.
- **Do not** introduce network access; all DBE interactions are modeled as file/CLI I/O.

---

## 5. Completion Criteria

The implementation is considered complete when:

1. A campaign can be run via CLI using a campaign file and:
   - Stock Wycheproof results are produced in the correct schema format.
   - Extended vectors (AI/Quantum) can be loaded from disk and tested.

2. PQC test suites exist and produce properly structured PQC results.

3. A `CryptoAssuranceSummary` can be generated for:
   - A specific library + version.
   - With clear `green/amber/red` risk ratings.

4. All new functionality is covered by at least basic self-tests / CI jobs.

Use `docs/MLOPS_INTEGRATION_NOTES.md` for hints on how this will be used in a broader pipeline.
