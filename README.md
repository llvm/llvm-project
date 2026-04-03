# Mage Fork of the LLVM Compiler Infrastructure

This repository is the LLVM fork used by [Mage](https://github.com/leandrolcampos/mage).

Mage is a research project focused on developing microarchitecture-aware and portable algorithms for elementary functions on modern GPUs. It investigates trade-offs and derives optimization heuristics for lookup tables, polynomial evaluation, and precision extension across diverse GPU designs and under multiple accuracy profiles. It also explores whether GPU-optimized implementations can generalize beyond GPUs while ensuring bitwise-identical results across architectures.

## Branch model

This fork uses two long-lived branches:

- `mage-staging`
- `main`

### `mage-staging`

This is the default branch and the main working branch for research and development. It contains the LLVM baseline actively used for the project. Update it from `main` during the first week of each month, or when needed.

Typical update flow:

```bash
git switch main
git fetch upstream
git merge --ff-only upstream/main
git push origin main

git switch mage-staging
git merge main
git push origin mage-staging
```

### `main`

This branch is a clean mirror of `upstream/main`. It must not carry Mage-specific changes.

```bash
git switch main
git fetch upstream
git merge --ff-only upstream/main
git push origin main
```

## Short-lived local branches

Short-lived research and development branches must be created from `mage-staging`.

Naming convention:

* `mage/<topic>`

Typical flow:

```bash
git switch mage-staging
git pull --ff-only
git switch -c mage/<topic>
```

Use these branches for work that belongs to the Mage workflow, including prototypes, measurements, and exploratory patches.

## Upstream branches

Branches intended for upstream pull requests must be created from `main`.

Naming convention:

* `upstream/<topic>`

Typical flow:

```bash
git switch main
git pull --ff-only
git switch -c upstream/<topic>
```

These branches should stay focused, reviewable, and as clean as possible with respect to upstream LLVM.

## Important rule

Do not use the same branch both as:

* a local Mage branch
* an upstream contribution branch

If some work starts in a `mage/` branch and later becomes worth upstreaming, reproduce or cherry-pick the relevant changes in a clean `upstream/` branch created from `main`.
