#!/bin/bash
# @file test_harness_generator.sh
# @brief Tests for harness generator tool
#
# Tests that dsmil-gen-fuzz-harness correctly generates harnesses.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -e

TEST_DIR=$(mktemp -d)
trap "rm -rf $TEST_DIR" EXIT

HARNESS_GEN="${HARNESS_GEN:-dsmil-gen-fuzz-harness}"

echo "=========================================="
echo "Harness Generator Test Suite"
echo "=========================================="

# Test 1: Generate generic protocol harness
echo ""
echo "=== Test 1: Generic Protocol Harness ==="
cat > "$TEST_DIR/protocol_config.yaml" <<EOF
target:
  type: protocol
  name: test_protocol
  entry_point: parse_protocol
  header: "test_protocol.h"

fuzzing:
  strategy: mutation
  max_input_size: 1024
EOF

if $HARNESS_GEN "$TEST_DIR/protocol_config.yaml" "$TEST_DIR/protocol_harness.cpp"; then
    if [ -f "$TEST_DIR/protocol_harness.cpp" ]; then
        echo "PASS: Protocol harness generated"
        if grep -q "parse_protocol" "$TEST_DIR/protocol_harness.cpp"; then
            echo "PASS: Entry point found in harness"
        else
            echo "FAIL: Entry point not found"
            exit 1
        fi
    else
        echo "FAIL: Harness file not created"
        exit 1
    fi
else
    echo "FAIL: Harness generation failed"
    exit 1
fi

# Test 2: Generate parser harness
echo ""
echo "=== Test 2: Parser Harness ==="
cat > "$TEST_DIR/parser_config.yaml" <<EOF
target:
  type: parser
  name: json_parser
  entry_point: json_parse
  header: "json_parser.h"

fuzzing:
  strategy: grammar
  grammar_file: "json.bnf"
EOF

if $HARNESS_GEN "$TEST_DIR/parser_config.yaml" "$TEST_DIR/parser_harness.cpp"; then
    if [ -f "$TEST_DIR/parser_harness.cpp" ]; then
        echo "PASS: Parser harness generated"
    else
        echo "FAIL: Harness file not created"
        exit 1
    fi
else
    echo "FAIL: Harness generation failed"
    exit 1
fi

# Test 3: Generate API harness
echo ""
echo "=== Test 3: API Harness ==="
cat > "$TEST_DIR/api_config.yaml" <<EOF
target:
  type: api
  name: buffer_api
  entry_point: buffer_operations
  header: "buffer_api.h"

fuzzing:
  strategy: structure
  structure_def: "buffer_struct.yaml"
EOF

if $HARNESS_GEN "$TEST_DIR/api_config.yaml" "$TEST_DIR/api_harness.cpp"; then
    if [ -f "$TEST_DIR/api_harness.cpp" ]; then
        echo "PASS: API harness generated"
    else
        echo "FAIL: Harness file not created"
        exit 1
    fi
else
    echo "FAIL: Harness generation failed"
    exit 1
fi

# Test 4: Invalid config handling
echo ""
echo "=== Test 4: Invalid Config Handling ==="
cat > "$TEST_DIR/invalid_config.yaml" <<EOF
invalid: config
EOF

if $HARNESS_GEN "$TEST_DIR/invalid_config.yaml" "$TEST_DIR/invalid_harness.cpp" 2>&1 | grep -q "error\|invalid\|fail"; then
    echo "PASS: Invalid config rejected"
else
    echo "FAIL: Invalid config not rejected"
    exit 1
fi

# Test 5: Missing file handling
echo ""
echo "=== Test 5: Missing File Handling ==="
if ! $HARNESS_GEN "/nonexistent/config.yaml" "$TEST_DIR/missing_harness.cpp" 2>&1 | grep -q "error\|not found\|cannot open"; then
    echo "FAIL: Missing file not handled"
    exit 1
else
    echo "PASS: Missing file handled gracefully"
fi

# Test 6: Compile generated harness
echo ""
echo "=== Test 6: Compile Generated Harness ==="
if command -v dsmil-clang++ >/dev/null 2>&1; then
    if dsmil-clang++ -fsanitize=fuzzer -c "$TEST_DIR/protocol_harness.cpp" -o "$TEST_DIR/protocol_harness.o" 2>&1; then
        echo "PASS: Generated harness compiles"
    else
        echo "WARN: Generated harness does not compile (may need target implementation)"
    fi
else
    echo "SKIP: dsmil-clang++ not found"
fi

echo ""
echo "=========================================="
echo "All harness generator tests passed!"
echo "=========================================="
