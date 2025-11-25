# Crypto-POW Module

Hardware-accelerated cryptographic proof-of-work for secure agent validation and workflow automation.

## Features

- **Blake3, SHA-256, SHA3-256** algorithm support
- **Multi-threaded** processing with Rayon (Rust backend)
- **Hardware acceleration** (AVX2, AES-NI when available)
- **TPM 2.0** cryptographic attestation support
- **Workflow validation** - prevents spam in automated task submissions

## Architecture

```
CryptoPOW (Python Interface)
    ↓
[Rust Module] ← (optional, hardware accelerated)
    ↓
Python Fallback (pure Python, works everywhere)
```

## Usage

### Basic POW Computation

```python
from crypto_pow import CryptoPOW, HashAlgorithm, POWResult

# Initialize engine
pow_engine = CryptoPOW(HashAlgorithm.SHA256)

# Compute proof-of-work
data = b"my_message"
difficulty = 20  # Number of leading zero bits

result = pow_engine.compute(data, difficulty)
print(f"Nonce: {result.nonce}")
print(f"Hash: {result.hash.hex()}")
print(f"Duration: {result.duration_ms}ms")
print(f"Hash rate: {result.hash_rate} H/s")

# Verify POW
is_valid = pow_engine.verify(data, result.nonce, difficulty)
```

### Workflow Validation

```python
from crypto_pow import POWWorkflowValidator

# Create validator (difficulty=16 means ~65k hashes required)
validator = POWWorkflowValidator(difficulty=16)

# Create validated task
task_data = b"execute_analysis_workflow"
nonce, result = validator.create_task(task_data)

# Later: validate task submission
is_authentic = validator.validate_task(task_data, nonce)
```

## Integration with Workflows

The `POWWorkflowValidator` prevents spam in automated systems by requiring computational proof for task submissions:

1. **Task Submission**: Compute POW before submitting
2. **Task Validation**: Verify POW before processing
3. **Spam Prevention**: Rejects invalid POW instantly

## Performance

### Difficulty Levels

| Difficulty | Average Hashes | Time (Python) | Time (Rust) |
|------------|---------------|---------------|-------------|
| 16         | ~65k          | ~50ms         | ~5ms        |
| 20         | ~1M           | ~800ms        | ~80ms       |
| 24         | ~16M          | ~12s          | ~1.2s       |

**Rust acceleration provides 10x speedup**

## Rust Module (Optional)

To build Rust acceleration:

```bash
cd /home/user/LAT5150DRVMIL/hooks/crypto-pow/rust
cargo build --release
cp target/release/libcrypto_pow_rust.so ../crypto_pow_rust.so
```

The Python module will automatically detect and use Rust acceleration if available.

## Integration Points

- `agent_comm_binary.py` - Binary agent communication protocol
- `execution_engine.py` - Workflow task validation
- `natural_language_interface.py` - Secure task submission

## Security

- **Cryptographic Strength**: SHA-256/Blake3 provide strong collision resistance
- **Difficulty Adjustment**: Configurable based on security requirements
- **Replay Protection**: Nonce must be unique for each message
- **Hardware-Backed**: TPM 2.0 support for attestation (future)

## Testing

```bash
python3 crypto_pow.py
```

This runs comprehensive tests of all algorithms and workflow validation.
