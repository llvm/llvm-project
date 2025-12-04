# DSLLVM Auto-Fuzz CI/CD Integration Guide

**Version:** 1.3.0
**Feature:** Auto-Generated Fuzz & Chaos Harnesses (Phase 1, Feature 1.2)
**SPDX-License-Identifier:** Apache-2.0 WITH LLVM-exception

## Overview

This guide covers integrating DSLLVM's automatic fuzz harness generation into CI/CD pipelines for continuous security testing. Key benefits:

- **Automatic fuzz target detection** via `DSMIL_UNTRUSTED_INPUT` annotations
- **Zero-config harness generation** using `dsmil-fuzz-gen`
- **Priority-based testing** focusing on high-risk functions first
- **Parallel fuzzing** across multiple CI runners
- **Corpus management** with automatic minimization
- **Crash reporting** integrated into PR workflows

## Architecture

```
┌─────────────────┐
│   Source Code   │
│ (with DSMIL_    │
│ UNTRUSTED_INPUT)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ dsmil-clang     │
│ -fdsmil-fuzz-   │
│  export         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ .dsmilfuzz.json │
│ (Fuzz Schema)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ dsmil-fuzz-gen  │
│ (L7 LLM)        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Fuzz Harnesses  │
│ (libFuzzer/AFL++)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ CI/CD Pipeline  │
│ (Parallel Fuzz) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Crash Reports   │
│ + Corpus        │
└─────────────────┘
```

## Quick Start

### 1. Add Untrusted Input Annotations

```c
#include <dsmil_attributes.h>

// Mark functions that process untrusted data
DSMIL_UNTRUSTED_INPUT
void parse_network_packet(const uint8_t *data, size_t len) {
    // Auto-fuzz will generate harness for this function
}

DSMIL_UNTRUSTED_INPUT
int parse_json(const char *json, size_t len, struct json_obj *out) {
    // Another fuzz target
}
```

### 2. Enable Fuzz Export in Build

```bash
# Add to your build script
dsmil-clang -fdsmil-fuzz-export src/*.c -o app
# This generates: app.dsmilfuzz.json
```

### 3. Copy CI/CD Template

```bash
# GitLab CI
# Use dynamic path resolution (relative to project root or DSMIL_DATA_DIR)
cp ${DSMIL_DATA_DIR:-dsmil}/tools/dsmil-fuzz-gen/ci-templates/gitlab-ci.yml .gitlab-ci.yml

# GitHub Actions
cp ${DSMIL_DATA_DIR:-dsmil}/tools/dsmil-fuzz-gen/ci-templates/github-actions.yml \
   .github/workflows/dsllvm-fuzz.yml
```

### 4. Commit and Push

```bash
git add .gitlab-ci.yml  # or .github/workflows/dsllvm-fuzz.yml
git commit -m "Add DSLLVM auto-fuzz CI/CD integration"
git push
```

CI/CD will automatically:
- Build with fuzz export
- Generate harnesses
- Run fuzzing on all targets
- Report crashes in PR comments

## Platform-Specific Integration

### GitLab CI

**Template:** `${DSMIL_DATA_DIR:-dsmil}/tools/dsmil-fuzz-gen/ci-templates/gitlab-ci.yml` (or use `dsmil_resolve_config()` API)

#### Pipeline Stages

1. **build:fuzz** - Compile with `-fdsmil-fuzz-export`
2. **fuzz:analyze** - Analyze fuzz targets and priorities
3. **fuzz:generate** - Generate and compile harnesses
4. **fuzz:test:quick** - Run quick fuzz tests (1 hour per target)
5. **fuzz:test:high_priority** - Extended fuzzing for high-risk targets
6. **report:fuzz** - Generate markdown report

#### Configuration

```yaml
variables:
  DSMIL_MISSION_PROFILE: "cyber_defence"
  FUZZ_TIMEOUT: "3600"  # 1 hour per target
  FUZZ_MAX_LEN: "65536"  # Max input size
```

#### Running Specific Stages

```bash
# Run only quick fuzz tests
gitlab-runner exec docker fuzz:test:quick

# Run nightly extended fuzzing
gitlab-runner exec docker fuzz:nightly
```

#### Artifacts

- `.dsmilfuzz.json` - Fuzz schemas (1 day)
- `fuzz_harnesses/` - Compiled harnesses (1 day)
- `crashes/` - Crash artifacts (30 days)
- `fuzz_corpus/` - Test corpus (90 days for nightly)
- `fuzz_report.md` - HTML report (30 days)

### GitHub Actions

**Template:** `${DSMIL_DATA_DIR:-dsmil}/tools/dsmil-fuzz-gen/ci-templates/github-actions.yml` (or use `dsmil_resolve_config()` API)

#### Workflow Jobs

1. **build-with-fuzz-export** - Build and generate schema
2. **generate-harnesses** - Create fuzz harnesses
3. **fuzz-test-quick** - Parallel quick fuzzing (4 shards)
4. **fuzz-test-high-priority** - Extended fuzzing (main branch only)
5. **report** - Generate report and comment on PRs
6. **corpus-management** - Merge and minimize corpus (main only)

#### Configuration

```yaml
env:
  DSMIL_MISSION_PROFILE: cyber_defence
  FUZZ_TIMEOUT: 3600
  FUZZ_MAX_LEN: 65536
```

#### Parallel Fuzzing

GitHub Actions runs 4 parallel fuzz shards by default:

```yaml
strategy:
  matrix:
    shard: [1, 2, 3, 4]
```

Adjust for more/less parallelism.

#### Scheduled Runs

```yaml
on:
  schedule:
    # Run nightly at 2 AM UTC
    - cron: '0 2 * * *'
```

#### PR Comments

Automatic PR comments with fuzz results:

```markdown
# DSLLVM Fuzz Test Report

**Date:** 2026-01-15T14:30:00Z
**Branch:** feature/new-parser
**Commit:** a1b2c3d4

## Fuzz Targets
- **parse_network_packet**: high priority (risk: 0.87)
- **parse_json**: medium priority (risk: 0.65)

## Results
- **Total Crashes:** 0
✅ No crashes found!
```

### Jenkins

#### Jenkinsfile Example

```groovy
pipeline {
    agent {
        docker {
            image 'dsllvm/toolchain:1.3.0'
        }
    }

    environment {
        DSMIL_MISSION_PROFILE = 'cyber_defence'
        FUZZ_TIMEOUT = '3600'
    }

    stages {
        stage('Build with Fuzz Export') {
            steps {
                sh '''
                    dsmil-clang -fdsmil-fuzz-export \
                        -fdsmil-mission-profile=${DSMIL_MISSION_PROFILE} \
                        src/*.c -o app
                '''
                archiveArtifacts artifacts: '*.dsmilfuzz.json', fingerprint: true
            }
        }

        stage('Generate Harnesses') {
            steps {
                sh '''
                    mkdir -p fuzz_harnesses
                    for schema in *.dsmilfuzz.json; do
                        dsmil-fuzz-gen "$schema" -o fuzz_harnesses/
                    done

                    cd fuzz_harnesses
                    for harness in *_fuzz.cpp; do
                        clang++ -fsanitize=fuzzer,address \
                            "$harness" ../app -o "${harness%.cpp}"
                    done
                '''
            }
        }

        stage('Run Fuzzing') {
            parallel {
                stage('Quick Fuzz') {
                    steps {
                        sh '''
                            cd fuzz_harnesses
                            mkdir -p ../crashes
                            for fuzz_bin in *_fuzz; do
                                timeout ${FUZZ_TIMEOUT} "./$fuzz_bin" \
                                    -max_total_time=${FUZZ_TIMEOUT} \
                                    -artifact_prefix=../crashes/ || true
                            done
                        '''
                    }
                }
                stage('High Priority') {
                    when {
                        branch 'main'
                    }
                    steps {
                        sh '''
                            jq -r '.fuzz_targets[] | select(.priority == "high") | .function' \
                                *.dsmilfuzz.json > high_priority.txt
                            cd fuzz_harnesses
                            while read target; do
                                "./${target}_fuzz" \
                                    -max_total_time=$((FUZZ_TIMEOUT * 3)) || true
                            done < ../high_priority.txt
                        '''
                    }
                }
            }
        }

        stage('Report') {
            steps {
                sh '''
                    crash_count=$(ls -1 crashes/ 2>/dev/null | wc -l)
                    echo "Crashes found: $crash_count"
                    if [ "$crash_count" -gt 0 ]; then
                        exit 1
                    fi
                '''
                publishHTML([
                    reportDir: 'crashes',
                    reportFiles: '*',
                    reportName: 'Fuzz Crashes'
                ])
            }
        }
    }

    post {
        always {
            archiveArtifacts artifacts: 'crashes/**', allowEmptyArchive: true
            archiveArtifacts artifacts: 'fuzz_harnesses/**', allowEmptyArchive: true
        }
    }
}
```

### CircleCI

#### .circleci/config.yml

```yaml
version: 2.1

orbs:
  dsllvm: dsllvm/auto-fuzz@1.3.0

jobs:
  build_and_fuzz:
    docker:
      - image: dsllvm/toolchain:1.3.0
    environment:
      DSMIL_MISSION_PROFILE: cyber_defence
      FUZZ_TIMEOUT: 3600
    steps:
      - checkout
      - run:
          name: Build with fuzz export
          command: |
            dsmil-clang -fdsmil-fuzz-export src/*.c -o app
      - run:
          name: Generate harnesses
          command: |
            mkdir fuzz_harnesses
            dsmil-fuzz-gen *.dsmilfuzz.json -o fuzz_harnesses/
            cd fuzz_harnesses && make
      - run:
          name: Run fuzzing
          command: |
            cd fuzz_harnesses
            for fuzz in *_fuzz; do
              timeout $FUZZ_TIMEOUT ./$fuzz \
                -max_total_time=$FUZZ_TIMEOUT || true
            done
      - store_artifacts:
          path: crashes/

workflows:
  version: 2
  fuzz_test:
    jobs:
      - build_and_fuzz
```

## Advanced Configuration

### Prioritized Fuzzing Strategy

Focus fuzzing effort on high-risk targets:

```bash
# Extract targets by priority
jq -r '.fuzz_targets[] | select(.l8_risk_score >= 0.7) | .function' \
    app.dsmilfuzz.json > high_risk.txt

# Allocate more time to high-risk targets
while read target; do
    timeout 7200 "./${target}_fuzz" -max_total_time=7200
done < high_risk.txt
```

### Corpus Management

#### Initial Seed Corpus

```bash
# Create seed corpus from test cases
mkdir -p seeds/parse_network_packet_fuzz
cp tests/packets/*.bin seeds/parse_network_packet_fuzz/

# Run with seeds
./parse_network_packet_fuzz seeds/parse_network_packet_fuzz/
```

#### Corpus Minimization

```bash
# Minimize corpus after fuzzing
./parse_network_packet_fuzz \
    -merge=1 -minimize_crash=1 \
    corpus_minimized/ corpus_raw/
```

#### Corpus Archiving

```yaml
# GitLab CI artifact
artifacts:
  paths:
    - fuzz_corpus/
  expire_in: 90 days
  when: always
```

```yaml
# GitHub Actions cache
- uses: actions/cache@v3
  with:
    path: fuzz_corpus/
    key: fuzz-corpus-${{ github.sha }}
    restore-keys: fuzz-corpus-
```

### Resource Limits

```bash
# Memory limit (2GB)
ulimit -v 2097152

# CPU time limit (1 hour)
ulimit -t 3600

# Core dumps disabled
ulimit -c 0

# Run with limits
./fuzz_harness -rss_limit_mb=2048 -timeout=30
```

### Parallel Fuzzing

#### GNU Parallel

```bash
# Fuzz all targets in parallel
ls -1 *_fuzz | parallel -j4 \
    'timeout 3600 {} -max_total_time=3600 -artifact_prefix=crashes/{/}_'
```

#### Docker Compose

```yaml
version: '3.8'
services:
  fuzz1:
    image: dsllvm/toolchain:1.3.0
    command: ./parse_packet_fuzz -max_total_time=3600
    volumes:
      - ./crashes:/crashes
  fuzz2:
    image: dsllvm/toolchain:1.3.0
    command: ./parse_json_fuzz -max_total_time=3600
    volumes:
      - ./crashes:/crashes
```

```bash
docker-compose up --abort-on-container-exit
```

## Crash Triage

### Automatic Deduplication

```bash
# libFuzzer automatic deduplication
./fuzz_harness \
    -exact_artifact_path=crash.bin \
    -minimize_crash=1 \
    crash.bin

# AFL++ deduplication
afl-tmin -i crashes/ -o crashes_unique/
```

### Crash Reporting

#### Create Crash Report

```bash
cat > crash_report.md <<EOF
# Crash Report: parse_network_packet

**Function:** parse_network_packet
**Priority:** high (L8 risk: 0.87)
**Sanitizer:** AddressSanitizer

## Stack Trace
\`\`\`
$(cat crash.log | head -50)
\`\`\`

## Input
\`\`\`
$(xxd crash.bin | head -20)
\`\`\`

## Reproducer
\`\`\`bash
./parse_network_packet_fuzz crash.bin
\`\`\`
EOF
```

#### Attach to Issue

```bash
# GitHub CLI
gh issue create \
    --title "Fuzz crash in parse_network_packet" \
    --body-file crash_report.md \
    --label "security,fuzzing,crash"

# GitLab API
curl -X POST \
    -H "PRIVATE-TOKEN: $GITLAB_TOKEN" \
    -F "title=Fuzz crash" \
    -F "description@crash_report.md" \
    "$GITLAB_URL/api/v4/projects/$PROJECT_ID/issues"
```

## Performance Optimization

### Fuzzing with Coverage Feedback

```bash
# Collect coverage
LLVM_PROFILE_FILE="fuzz-%p.profraw" \
    ./fuzz_harness -runs=100000

# Merge coverage
llvm-profdata merge -sparse fuzz-*.profraw -o fuzz.profdata

# Visualize coverage
llvm-cov show ./fuzz_harness \
    -instr-profile=fuzz.profdata \
    -format=html -output-dir=coverage/
```

### Distributed Fuzzing

```bash
# Coordinator node
./fuzz_harness \
    -jobs=0 \
    -workers=4 \
    -artifact_prefix=crashes/ \
    corpus/

# Worker nodes (connect to coordinator)
./fuzz_harness \
    -fork=4 \
    -artifact_prefix=crashes/ \
    corpus/
```

## Integration with Security Tools

### OSS-Fuzz Integration

```python
# projects/myproject/build.sh
cd $SRC/myproject
dsmil-clang -fdsmil-fuzz-export src/*.c
dsmil-fuzz-gen *.dsmilfuzz.json -o $OUT/

cd $OUT
for harness in *_fuzz.cpp; do
    $CXX $CXXFLAGS $LIB_FUZZING_ENGINE \
        "$harness" -o "${harness%.cpp}"
done
```

### Fuzzbench Integration

```yaml
# myproject.yaml
fuzzer: libfuzzer
build_script: |
  dsmil-clang -fdsmil-fuzz-export
  dsmil-fuzz-gen --fuzzer=libFuzzer
targets:
  - parse_network_packet_fuzz
  - parse_json_fuzz
```

## Troubleshooting

### No Fuzz Targets Generated

**Problem:** `.dsmilfuzz.json` is empty or missing

**Solution:**
1. Check for `DSMIL_UNTRUSTED_INPUT` annotations
2. Verify `-fdsmil-fuzz-export` flag is used
3. Ensure functions have untrusted input parameters

```c
// Add annotation
DSMIL_UNTRUSTED_INPUT
void my_parser(const uint8_t *data, size_t len);
```

### Harness Compilation Fails

**Problem:** Linker errors when compiling harness

**Solution:**
```bash
# Include all necessary objects
clang++ -fsanitize=fuzzer \
    harness.cpp \
    libmyapp.a \
    -lpthread -lm
```

### Fuzzing Too Slow

**Problem:** Low exec/s rate

**Solution:**
1. Reduce input size: `-max_len=1024`
2. Enable parallel fuzzing: `-jobs=4 -workers=4`
3. Use faster sanitizer: `-fsanitize=fuzzer` only (no ASan)
4. Profile and optimize target function

### OOM Crashes

**Problem:** Out-of-memory errors

**Solution:**
```bash
# Set RSS limit
./fuzz_harness -rss_limit_mb=2048

# Reduce max input
./fuzz_harness -max_len=8192
```

## Best Practices

### 1. Annotate All Untrusted Inputs

```c
// ✓ GOOD
DSMIL_UNTRUSTED_INPUT
void parse_request(const uint8_t *buf, size_t len);

// ✗ BAD - Missing annotation
void parse_request(const uint8_t *buf, size_t len);
```

### 2. Start with Quick Runs

```bash
# Initial run (5 minutes)
timeout 300 ./fuzz_harness -max_total_time=300

# Gradually increase time
timeout 3600 ./fuzz_harness -max_total_time=3600
```

### 3. Use Mission Profiles

```bash
# Development: permissive
dsmil-clang -fdsmil-mission-profile=lab_research ...

# CI: stricter
dsmil-clang -fdsmil-mission-profile=cyber_defence ...

# Production: strictest
dsmil-clang -fdsmil-mission-profile=border_ops ...
```

### 4. Archive Crashes

```bash
# Tag crashes with commit hash
mkdir -p crashes/${CI_COMMIT_SHA}
mv crash-* crashes/${CI_COMMIT_SHA}/
```

### 5. Regression Test

```bash
# Run corpus regression on every PR
for corpus in corpus/*/; do
    ./fuzz_harness "$corpus" -runs=0 || exit 1
done
```

## References

- **Fuzz Export Pass:** `dsmil/lib/Passes/DsmilFuzzExportPass.cpp`
- **Schema Specification:** `dsmil/docs/FUZZ-HARNESS-SCHEMA.md`
- **dsmil-fuzz-gen:** `dsmil/tools/dsmil-fuzz-gen/README.md`
- **Mission Profiles:** `dsmil/docs/MISSION-PROFILES-GUIDE.md`
- **DSLLVM Roadmap:** `dsmil/docs/DSLLVM-ROADMAP.md`

## Support

- Issues: https://github.com/dsllvm/dsllvm/issues
- Documentation: https://dsmil.org/docs/auto-fuzz
- Mailing List: dsllvm-security@lists.llvm.org
