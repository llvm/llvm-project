# MLOps Integration Notes – DSMIL–Wycheproof

These notes explain how this Wycheproof fork will be used inside the broader DSMIL
AI/MLOps pipeline.

## 1. Conceptual Mapping

- Each **crypto implementation variant** is treated like a **model version**:
  - `(lib_id, lib_version, build_id, compiler, hardware_profile)`.
- A **Wycheproof campaign** behaves like a validation run:
  - Inputs: campaign definition.
  - Outputs: test results + crypto assurance summary.

## 2. Required Interfaces

An external MLOps pipeline will:

1. Generate a campaign config (JSON/YAML).
2. Call a CLI/library entrypoint in this repo.
3. Receive:
   - `crypto_test_result` records (for detailed logs).
   - A `crypto_assurance_summary` (for gate decisions).

You must:

- Keep the CLI non-interactive.
- Make failures deterministic and machine-readable.

## 3. Gating Logic (Example)

A deployment pipeline may block a crypto build if:

- Any primitive used in production has:
  - `risk_rating = "red"` for that build; or
  - `failed + errors > threshold`.

You do not need to implement the pipeline itself; only ensure that:

- Results and summaries are emitted in the schemas described in `schemas/*.schema.yaml`.
- Tooling is simple to call from automation (CI/CD or external MLOps systems).

## 4. Output Locations

For simplicity, default to:

- Results: `./out/results/<campaign_id>.json`
- Summaries: `./out/summaries/<lib_id>-<lib_version>-<timestamp>.json`

Allow overriding paths via CLI flags or environment variables.

These conventions will be assumed by higher-level automation unless configured otherwise.
