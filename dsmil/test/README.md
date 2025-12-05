# DSLLVM Test Suite

Comprehensive test suite for DSLLVM covering all implemented features.

## Test Structure

```
test/
├── runtime/              # Unit tests for runtime libraries
│   ├── test_ot_telemetry.c
│   ├── test_fuzz_telemetry.c
│   └── test_fuzz_telemetry_advanced.c
├── passes/               # LLVM pass integration tests
│   ├── test_ot_telemetry_pass.c
│   ├── test_telecom_pass.c
│   └── test_fuzz_coverage_pass.c
├── integration/          # Integration tests
│   ├── test_telecom_macros.c
│   ├── test_attributes.c
│   └── test_end_to_end.c
└── CMakeLists.txt        # Build configuration
```

## Running Tests

### Build Tests

```bash
cd build
cmake -DCMAKE_BUILD_TYPE=Debug ..
make test_ot_telemetry_runtime
make test_fuzz_telemetry_runtime
make test_fuzz_telemetry_advanced_runtime
make test_telecom_macros
make test_attributes
make test_end_to_end
```

### Run All Runtime Tests

```bash
make check-dsmil-runtime
```

### Run Individual Tests

```bash
./test_ot_telemetry_runtime
./test_fuzz_telemetry_runtime
./test_fuzz_telemetry_advanced_runtime
./test_telecom_macros
./test_attributes
./test_end_to_end
```

### Run LLVM Pass Tests (LIT)

```bash
llvm-lit test/passes/ -v
```

## Test Coverage

### Runtime Tests

- **OT Telemetry Runtime** (`test_ot_telemetry.c`):
  - ✅ Initialization and shutdown
  - ✅ Environment variable handling
  - ✅ All event types
  - ✅ Safety signal updates
  - ✅ Null event handling
  - ✅ Ring buffer operations
  - ✅ Complete event logging

- **Fuzzing Telemetry Runtime** (`test_fuzz_telemetry.c`):
  - ✅ Initialization and shutdown
  - ✅ Context management
  - ✅ Coverage tracking
  - ✅ State machine transitions
  - ✅ Metrics collection
  - ✅ API misuse detection
  - ✅ Event export
  - ✅ Budget checking
  - ✅ Ring buffer overflow handling

- **Advanced Fuzzing Telemetry Runtime** (`test_fuzz_telemetry_advanced.c`):
  - ✅ Advanced initialization
  - ✅ Coverage map operations
  - ✅ ML integration stubs
  - ✅ Performance counters
  - ✅ Statistics collection
  - ✅ Advanced event export

### Pass Tests

- **OT Telemetry Pass** (`test_ot_telemetry_pass.c`):
  - ✅ OT-critical function instrumentation
  - ✅ SES gate instrumentation
  - ✅ Safety signal instrumentation
  - ✅ Manifest generation

- **Telecom Pass** (`test_telecom_pass.c`):
  - ✅ SS7 annotation discovery
  - ✅ SIGTRAN annotation discovery
  - ✅ Environment classification
  - ✅ Manifest generation

- **Fuzzing Coverage Pass** (`test_fuzz_coverage_pass.c`):
  - ✅ Coverage site instrumentation
  - ✅ State machine instrumentation
  - ✅ Critical operation tracking
  - ✅ API misuse detection

### Integration Tests

- **Telecom Macros** (`test_telecom_macros.c`):
  - ✅ SS7 RX/TX macros
  - ✅ SIGTRAN RX/TX macros
  - ✅ Signal anomaly macro
  - ✅ SS7 full macro

- **Attributes** (`test_attributes.c`):
  - ✅ All OT attributes compile
  - ✅ All telecom attributes compile
  - ✅ All fuzzing attributes compile
  - ✅ All security attributes compile

- **End-to-End** (`test_end_to_end.c`):
  - ✅ Complete OT workflow
  - ✅ Complete telecom workflow
  - ✅ Complete fuzzing workflow
  - ✅ Combined workflows

## Coverage Goals

- **Runtime Libraries**: 100% line coverage
- **LLVM Passes**: 100% line coverage
- **Integration**: All workflows tested

## Measuring Coverage

```bash
# Build with coverage
cmake -DCMAKE_BUILD_TYPE=Coverage \
      -DCMAKE_C_FLAGS="--coverage" \
      -DCMAKE_CXX_FLAGS="--coverage" \
      ..

make

# Run tests
make check-dsmil-runtime

# Generate report
make coverage
# or
lcov --capture --directory . --output-file coverage.info
genhtml coverage.info --output-directory coverage
```

## Adding New Tests

1. **Runtime Tests**: Add to `test/runtime/`
2. **Pass Tests**: Add to `test/passes/` (use LIT format)
3. **Integration Tests**: Add to `test/integration/`

Follow existing test patterns and ensure:
- Clear test names
- Comprehensive coverage
- Error case handling
- Edge case testing
