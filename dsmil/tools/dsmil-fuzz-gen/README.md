# dsmil-fuzz-gen - DSLLVM Fuzz Harness Generator

**Version:** 1.3.0
**Feature:** Auto-Generated Fuzz & Chaos Harnesses (Phase 1, Feature 1.2)

## Overview

`dsmil-fuzz-gen` is a tool that automatically generates fuzzing harnesses from `.dsmilfuzz.json` schema files produced by the DSLLVM `DsmilFuzzExportPass`. It supports:

- **libFuzzer** harness generation
- **AFL++** harness generation
- **Layer 7 LLM** integration for AI-assisted harness generation
- **Offline mode** with template-based generation

## Installation

```bash
# Add to PATH
export PATH="/home/user/DSLLVM/dsmil/tools/dsmil-fuzz-gen:$PATH"

# Optional: Install Python dependencies for L7 LLM integration
pip install requests
```

## Usage

### Basic Usage

```bash
# Generate libFuzzer harness (default)
dsmil-fuzz-gen network_daemon.dsmilfuzz.json

# Generate AFL++ harness
dsmil-fuzz-gen network_daemon.dsmilfuzz.json --fuzzer=AFL++

# Specify output directory
dsmil-fuzz-gen network_daemon.dsmilfuzz.json -o fuzz_harnesses/
```

### With Layer 7 LLM

```bash
# Configure L7 LLM service URL
export DSMIL_L7_LLM_URL=http://layer7-llm.local:8080/api/v1

# Generate harnesses with AI assistance
dsmil-fuzz-gen network_daemon.dsmilfuzz.json
```

### Offline Mode

```bash
# Use template-based generation (no network required)
dsmil-fuzz-gen network_daemon.dsmilfuzz.json --offline
```

## Complete Workflow

### Step 1: Compile with Fuzz Export

```bash
dsmil-clang -fdsmil-fuzz-export src/network.c -o network_daemon
# Output: network_daemon.dsmilfuzz.json
```

### Step 2: Generate Harness

```bash
dsmil-fuzz-gen network_daemon.dsmilfuzz.json --fuzzer=libFuzzer -o fuzz/
# Output: fuzz/parse_network_packet_fuzz.cpp
```

### Step 3: Compile Harness

```bash
cd fuzz/
clang++ -fsanitize=fuzzer,address,undefined -g -O1 \\
  parse_network_packet_fuzz.cpp ../network_daemon.o \\
  -o parse_network_packet_fuzz
```

### Step 4: Run Fuzzer

```bash
# Create seed corpus
mkdir seeds
echo "GET / HTTP/1.1" > seeds/seed1.txt

# Run libFuzzer
./parse_network_packet_fuzz -max_total_time=3600 -print_final_stats=1 seeds/
```

## Generated Harness Structure

### libFuzzer Harness

```cpp
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    // Size constraints from parameter_domains
    if (size > 65535) return 0;

    // Call fuzz target
    parse_network_packet(data, size);

    return 0;
}
```

### AFL++ Harness

```cpp
int main(int argc, char **argv) {
    uint8_t buffer[65536];
    size_t size = fread(buffer, 1, sizeof(buffer), stdin);

    if (size == 0) return 0;

    parse_network_packet(buffer, size);

    return 0;
}
```

## Integration with CI/CD

### GitLab CI

```yaml
fuzz_test:
  stage: security
  script:
    - dsmil-clang -fdsmil-fuzz-export src/*.c -o app
    - dsmil-fuzz-gen app.dsmilfuzz.json -o fuzz/
    - cd fuzz && ./build_harnesses.sh
    - ./run_fuzzers.sh --timeout=3600
  artifacts:
    paths: [crash-*, leak-*]
```

### GitHub Actions

```yaml
- name: Generate Fuzz Harnesses
  run: |
    dsmil-clang -fdsmil-fuzz-export src/*.c -o app
    dsmil-fuzz-gen app.dsmilfuzz.json

- name: Compile and Run Fuzzing
  run: |
    clang++ -fsanitize=fuzzer,address *_fuzz.cpp -o fuzz_harness
    ./fuzz_harness -max_total_time=3600
```

## Layer 7 LLM Integration

When Layer 7 LLM is available, `dsmil-fuzz-gen` can leverage AI assistance for:

1. **Smart Parameter Extraction**: AI understands complex parameter relationships
2. **Edge Case Generation**: AI suggests interesting edge cases to test
3. **Custom Constraints**: AI translates semantic constraints into code

### Example L7 LLM Request

```json
{
  "prompt": "Generate libFuzzer harness for function 'parse_json' with parameters: json_str (bytes), length (size_t). The json_str must be UTF-8 encoded and length <= 1048576.",
  "model": "dsmil-l7-llm-v1",
  "temperature": 0.2
}
```

### Example L7 LLM Response

```cpp
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size > 1048576) return 0;  // Max 1MB
    if (size == 0) return 0;

    // Validate UTF-8 (basic check)
    for (size_t i = 0; i < size; i++) {
        if (data[i] == 0) return 0;  // No embedded nulls
    }

    struct json_object out;
    parse_json((const char*)data, size, &out);

    return 0;
}
```

## Command-Line Options

```
usage: dsmil-fuzz-gen [-h] [--fuzzer {libFuzzer,AFL++,Honggfuzz}]
                      [--output OUTPUT] [--l7-llm-url L7_LLM_URL]
                      [--offline] [--version]
                      schema

positional arguments:
  schema                Path to .dsmilfuzz.json schema file

optional arguments:
  -h, --help            show this help message and exit
  --fuzzer {libFuzzer,AFL++,Honggfuzz}, -f {libFuzzer,AFL++,Honggfuzz}
                        Target fuzzing engine (default: libFuzzer)
  --output OUTPUT, -o OUTPUT
                        Output directory for harness files (default: .)
  --l7-llm-url L7_LLM_URL
                        Layer 7 LLM service URL (default: $DSMIL_L7_LLM_URL)
  --offline             Offline mode (no L7 LLM queries)
  --version, -v         show program's version number and exit
```

## Environment Variables

- `DSMIL_L7_LLM_URL`: Layer 7 LLM service URL for AI-assisted generation
  - Example: `http://layer7-llm.local:8080/api/v1`

## Troubleshooting

### No Fuzz Targets Found

```
[WARNING] No fuzz targets found in schema
```

**Solution:** Ensure functions are annotated with `DSMIL_UNTRUSTED_INPUT`:

```c
DSMIL_UNTRUSTED_INPUT
void parse_packet(const uint8_t *data, size_t len);
```

### L7 LLM Connection Failed

```
[WARNING] L7 LLM query failed: Connection refused
```

**Solution:** Use offline mode or configure correct L7 LLM URL:

```bash
dsmil-fuzz-gen --offline network_daemon.dsmilfuzz.json
# OR
export DSMIL_L7_LLM_URL=http://correct-url:8080/api/v1
```

### Missing Python Requests Module

```bash
pip install requests
```

## Examples

See `dsmil/test/mission-profiles/cyber_defence_example.c` for a complete example with `DSMIL_UNTRUSTED_INPUT` annotations.

## References

- **Fuzz Export Pass:** `dsmil/lib/Passes/DsmilFuzzExportPass.cpp`
- **Schema Specification:** `dsmil/docs/FUZZ-HARNESS-SCHEMA.md`
- **DSLLVM Roadmap:** `dsmil/docs/DSLLVM-ROADMAP.md`
- **libFuzzer:** https://llvm.org/docs/LibFuzzer.html
- **AFL++:** https://github.com/AFLplusplus/AFLplusplus
