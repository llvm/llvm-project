#!/usr/bin/env python3
"""
Performance Benchmark: Binary Communication vs Pydantic Validation
Measures throughput, latency, and overhead for both approaches

Use cases:
- Binary: Ultra-low latency agent communication (microseconds matter)
- Pydantic: Type-safe API responses (milliseconds acceptable)

Author: DSMIL Integration Framework
"""

import time
import struct
import json
from dataclasses import dataclass
from typing import List
import statistics

try:
    from pydantic import BaseModel, Field
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    print("⚠️  Pydantic not available. Install with: pip install pydantic")


# ============================================================================
# Test Data Structures
# ============================================================================

@dataclass
class AgentResultDataclass:
    """Dataclass version (no validation)"""
    task_id: str
    agent_id: str
    success: bool
    content: str
    latency_ms: float
    model_used: str


if PYDANTIC_AVAILABLE:
    class AgentResultPydantic(BaseModel):
        """Pydantic version (with validation)"""
        task_id: str = Field(..., min_length=1, max_length=100)
        agent_id: str = Field(..., min_length=1, max_length=100)
        success: bool
        content: str = Field(..., min_length=1)
        latency_ms: float = Field(..., ge=0)
        model_used: str


# Binary format struct: task_id(32), agent_id(32), success(1), latency_ms(8), content_len(4), content(var)
BINARY_HEADER_FORMAT = '32s32s?d'  # task_id, agent_id, success, latency_ms
BINARY_HEADER_SIZE = struct.calcsize(BINARY_HEADER_FORMAT)


# ============================================================================
# Serialization Functions
# ============================================================================

def serialize_binary(result: AgentResultDataclass) -> bytes:
    """Serialize to binary format (fastest)"""
    content_bytes = result.content.encode('utf-8')
    model_bytes = result.model_used.encode('utf-8')

    # Pack header
    header = struct.pack(
        BINARY_HEADER_FORMAT,
        result.task_id.encode('utf-8').ljust(32, b'\x00'),
        result.agent_id.encode('utf-8').ljust(32, b'\x00'),
        result.success,
        result.latency_ms
    )

    # Pack variable-length data
    content_len = struct.pack('I', len(content_bytes))
    model_len = struct.pack('I', len(model_bytes))

    return header + content_len + content_bytes + model_len + model_bytes


def deserialize_binary(data: bytes) -> AgentResultDataclass:
    """Deserialize from binary format"""
    # Unpack header
    task_id_raw, agent_id_raw, success, latency_ms = struct.unpack(
        BINARY_HEADER_FORMAT,
        data[:BINARY_HEADER_SIZE]
    )

    offset = BINARY_HEADER_SIZE

    # Unpack content
    content_len = struct.unpack('I', data[offset:offset+4])[0]
    offset += 4
    content = data[offset:offset+content_len].decode('utf-8')
    offset += content_len

    # Unpack model
    model_len = struct.unpack('I', data[offset:offset+4])[0]
    offset += 4
    model_used = data[offset:offset+model_len].decode('utf-8')

    return AgentResultDataclass(
        task_id=task_id_raw.rstrip(b'\x00').decode('utf-8'),
        agent_id=agent_id_raw.rstrip(b'\x00').decode('utf-8'),
        success=success,
        content=content,
        latency_ms=latency_ms,
        model_used=model_used
    )


def serialize_json_dict(result: AgentResultDataclass) -> bytes:
    """Serialize to JSON dict (no validation)"""
    data = {
        'task_id': result.task_id,
        'agent_id': result.agent_id,
        'success': result.success,
        'content': result.content,
        'latency_ms': result.latency_ms,
        'model_used': result.model_used,
    }
    return json.dumps(data).encode('utf-8')


def deserialize_json_dict(data: bytes) -> AgentResultDataclass:
    """Deserialize from JSON dict"""
    obj = json.loads(data.decode('utf-8'))
    return AgentResultDataclass(**obj)


if PYDANTIC_AVAILABLE:
    def serialize_pydantic(result: AgentResultPydantic) -> bytes:
        """Serialize Pydantic model to JSON (with validation)"""
        return result.model_dump_json().encode('utf-8')

    def deserialize_pydantic(data: bytes) -> AgentResultPydantic:
        """Deserialize to Pydantic model (with validation)"""
        return AgentResultPydantic.model_validate_json(data)


# ============================================================================
# Benchmark Functions
# ============================================================================

def benchmark_method(
    name: str,
    serialize_func,
    deserialize_func,
    test_data: List[AgentResultDataclass],
    iterations: int = 10000
) -> dict:
    """Benchmark a serialization method"""
    print(f"\n{'='*70}")
    print(f"Benchmarking: {name}")
    print(f"{'='*70}")

    # Warmup
    for item in test_data[:100]:
        serialized = serialize_func(item)
        deserialize_func(serialized)

    # Serialize benchmark
    serialize_times = []
    for _ in range(iterations):
        for item in test_data:
            start = time.perf_counter()
            serialized = serialize_func(item)
            serialize_times.append((time.perf_counter() - start) * 1_000_000)  # microseconds

    # Deserialize benchmark
    serialized_data = [serialize_func(item) for item in test_data]
    deserialize_times = []
    for _ in range(iterations):
        for data in serialized_data:
            start = time.perf_counter()
            deserialize_func(data)
            deserialize_times.append((time.perf_counter() - start) * 1_000_000)  # microseconds

    # Size benchmark
    sizes = [len(serialize_func(item)) for item in test_data]

    results = {
        'name': name,
        'serialize_avg_us': statistics.mean(serialize_times),
        'serialize_median_us': statistics.median(serialize_times),
        'serialize_p99_us': sorted(serialize_times)[int(len(serialize_times) * 0.99)],
        'deserialize_avg_us': statistics.mean(deserialize_times),
        'deserialize_median_us': statistics.median(deserialize_times),
        'deserialize_p99_us': sorted(deserialize_times)[int(len(deserialize_times) * 0.99)],
        'avg_size_bytes': statistics.mean(sizes),
        'throughput_ops_sec': 1_000_000 / (statistics.mean(serialize_times) + statistics.mean(deserialize_times)),
    }

    print(f"Serialize:   {results['serialize_avg_us']:.2f}μs avg, {results['serialize_median_us']:.2f}μs median, {results['serialize_p99_us']:.2f}μs p99")
    print(f"Deserialize: {results['deserialize_avg_us']:.2f}μs avg, {results['deserialize_median_us']:.2f}μs median, {results['deserialize_p99_us']:.2f}μs p99")
    print(f"Size:        {results['avg_size_bytes']:.0f} bytes avg")
    print(f"Throughput:  {results['throughput_ops_sec']:.0f} ops/sec")

    return results


def generate_test_data(count: int = 100) -> List[AgentResultDataclass]:
    """Generate realistic test data"""
    test_data = []
    for i in range(count):
        test_data.append(AgentResultDataclass(
            task_id=f"task_{i:06d}",
            agent_id=f"agent_{i % 10:03d}",
            success=True,
            content=f"This is a test response with some content. " * 10,  # ~500 bytes
            latency_ms=float(i % 100 + 10),
            model_used="deepseek-coder:6.7b"
        ))
    return test_data


# ============================================================================
# Main Benchmark
# ============================================================================

def main():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║  Binary Communication vs Pydantic Validation - Performance Benchmark ║
╚══════════════════════════════════════════════════════════════════════╝
""")

    # Generate test data
    print("Generating test data...")
    test_data = generate_test_data(100)

    iterations = 1000
    print(f"Running {iterations} iterations per method...\n")

    results = []

    # Benchmark 1: Binary (fastest, no validation)
    results.append(benchmark_method(
        "Binary Communication (No Validation)",
        serialize_binary,
        deserialize_binary,
        test_data,
        iterations
    ))

    # Benchmark 2: JSON dict (no validation)
    results.append(benchmark_method(
        "JSON Dict (No Validation)",
        serialize_json_dict,
        deserialize_json_dict,
        test_data,
        iterations
    ))

    # Benchmark 3: Pydantic (with validation)
    if PYDANTIC_AVAILABLE:
        pydantic_data = [
            AgentResultPydantic(
                task_id=item.task_id,
                agent_id=item.agent_id,
                success=item.success,
                content=item.content,
                latency_ms=item.latency_ms,
                model_used=item.model_used
            ) for item in test_data
        ]
        results.append(benchmark_method(
            "Pydantic JSON (With Validation)",
            serialize_pydantic,
            deserialize_pydantic,
            pydantic_data,
            iterations
        ))

    # Summary comparison
    print(f"\n{'='*70}")
    print("SUMMARY - Relative Performance")
    print(f"{'='*70}\n")

    binary_result = results[0]
    print(f"{'Method':<40} {'Speed vs Binary':<20} {'Use Case'}")
    print(f"{'-'*70}")

    for result in results:
        speedup = binary_result['throughput_ops_sec'] / result['throughput_ops_sec']
        use_case = {
            'Binary Communication': 'Agent IPC, real-time data',
            'JSON Dict': 'Simple APIs, no validation needed',
            'Pydantic JSON': 'Type-safe APIs, validation required'
        }.get(result['name'], 'Unknown')

        print(f"{result['name']:<40} {speedup:.2f}x slower{'':<10} {use_case}")

    print(f"\n{'='*70}")
    print("RECOMMENDATIONS")
    print(f"{'='*70}\n")

    print("Use BINARY when:")
    print("  ✅ Ultra-low latency required (microseconds matter)")
    print("  ✅ Agent-to-agent IPC within same machine")
    print("  ✅ High-throughput real-time data streams")
    print("  ✅ Fixed schema, trusted source\n")

    print("Use PYDANTIC when:")
    print("  ✅ Type safety critical (prevent bugs)")
    print("  ✅ Web APIs with external clients")
    print("  ✅ Schema validation needed (security)")
    print("  ✅ Developer experience matters (IDE autocomplete)")
    print("  ✅ Millisecond latency acceptable\n")

    print("HYBRID APPROACH (Recommended):")
    print("  🔄 Binary for agent IPC (speed-critical path)")
    print("  🔄 Pydantic for web API + CLI (type safety + UX)")
    print("  🔄 Best of both worlds!")


if __name__ == "__main__":
    main()
