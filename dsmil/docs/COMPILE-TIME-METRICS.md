# Compile-Time Performance Metrics Guide

**DSLLVM v1.7+ Compile-Time Metrics**

## Overview

DSLLVM collects comprehensive metrics during compilation to provide visibility into build performance, pass execution time, and optimization effectiveness.

---

## Quick Start

```bash
# Enable metrics collection
dsmil-clang -fdsmil-metrics -fdsmil-metrics-output=build.json -O3 input.c

# View metrics report
dsmil-metrics report build.json

# Compare builds
dsmil-metrics compare build1.json build2.json

# Generate dashboard
dsmil-metrics dashboard build.json --output=dashboard.html
```

---

## Metrics Collected

### Pass Performance

- **Execution Time**: Time spent in each LLVM pass
- **Memory Usage**: Peak and average memory per pass
- **IR Transformation**: IR size before/after each pass

### Feature Impact

- **Stealth Mode Overhead**: Cost of stealth transformations
- **Classification Checks**: Time spent on security checks
- **Provenance Generation**: Time to generate and sign provenance
- **Threat Signature**: Time to embed threat signatures

### Optimization Effectiveness

- **Code Size Reduction**: IR size reduction percentage
- **Estimated Speedup**: Performance improvement estimates
- **Device Placement**: Optimization recommendations followed

---

## Metrics Report

View detailed metrics report:

```bash
dsmil-metrics report build.json
```

Output includes:
- Total compile time
- Peak memory usage
- Per-pass breakdown
- Feature overhead analysis
- Optimization effectiveness

---

## Build Comparison

Compare two builds to identify regressions:

```bash
dsmil-metrics compare build1.json build2.json
```

Shows:
- Time differences per pass
- Memory usage changes
- Optimization effectiveness changes
- Feature overhead differences

---

## HTML Dashboard

Generate interactive HTML dashboard:

```bash
dsmil-metrics dashboard build.json --output=dashboard.html
```

Dashboard includes:
- Visual charts and graphs
- Interactive pass breakdown
- Feature impact visualization
- Optimization effectiveness metrics

---

## CI/CD Integration

Integrate metrics collection into build pipelines:

```yaml
# GitHub Actions example
- name: Build with metrics
  run: |
    dsmil-clang -fdsmil-metrics -fdsmil-metrics-output=metrics.json -O3 src/*.c
    
- name: Upload metrics
  uses: actions/upload-artifact@v3
  with:
    name: build-metrics
    path: metrics.json
```

---

## API Usage

Use metrics API in custom passes:

```c
#include "dsmil_metrics.h"

int pass_id = dsmil_metrics_start_pass("MyPass");
// ... pass execution ...
dsmil_metrics_end_pass(pass_id, ir_size_before, ir_size_after);
```

---

## Related Documentation

- **[PIPELINES.md](PIPELINES.md)**: Pass pipeline configurations
- **[DSLLVM-DESIGN.md](DSLLVM-DESIGN.md)**: Complete design specification

---

**DSLLVM Compile-Time Metrics**: Data-driven optimization and performance analysis.
