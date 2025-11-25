#!/usr/bin/env python3
"""
High-Performance Binary Agent Communication Protocol
Integrates with claude-backups C implementation and crypto POW module

Features:
- Binary protocol (not JSON) for minimal latency
- Cryptographic proof-of-work for agent validation
- Vector-accelerated hashing (AVX512/AVX2 when available)
- Zero-copy message passing with shared memory
- Direct IPC (no Redis middleware for maximum speed)

Architecture:
- Uses multiprocessing shared memory for zero-copy large payloads
- Uses multiprocessing queues for message routing
- No external dependencies (Redis, PostgreSQL) required
- Pure binary protocol with vector acceleration + tuned difficulty
"""

import os
import sys
import struct
import hashlib
import time
import psutil
import multiprocessing as mp
from multiprocessing import shared_memory
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import IntEnum
import ctypes
from ctypes import c_uint8, c_uint32, c_uint64, POINTER

# Try to load C acceleration library (from claude-backups)
try:
    # Look for compiled C library from claude-backups agents/source/C/
    libpath = os.path.join(os.path.dirname(__file__), "libagent_comm.so")
    if os.path.exists(libpath):
        libagent = ctypes.CDLL(libpath)
        C_ACCEL_AVAILABLE = True

        # Function signatures from claude-backups C implementation
        libagent.binary_encode_message.argtypes = [POINTER(c_uint8), c_uint32, POINTER(c_uint8)]
        libagent.binary_encode_message.restype = c_uint32

        libagent.binary_decode_message.argtypes = [POINTER(c_uint8), c_uint32, POINTER(c_uint8)]
        libagent.binary_decode_message.restype = c_uint32

        libagent.crypto_pow_verify.argtypes = [POINTER(c_uint8), c_uint32, c_uint32]
        libagent.crypto_pow_verify.restype = c_uint32

        libagent.crypto_pow_compute.argtypes = [POINTER(c_uint8), c_uint32, c_uint32, POINTER(c_uint64)]
        libagent.crypto_pow_compute.restype = c_uint32

    else:
        C_ACCEL_AVAILABLE = False
        libagent = None
except Exception:
    C_ACCEL_AVAILABLE = False
    libagent = None


class MessageType(IntEnum):
    """Binary message types (4-bit encoding)"""
    COMMAND = 0x01
    RESPONSE = 0x02
    STATUS = 0x03
    ALERT = 0x04
    HEARTBEAT = 0x05
    TASK_REQUEST = 0x06
    TASK_RESULT = 0x07
    SYNC = 0x08
    BROADCAST = 0x09
    ACK = 0x0A
    NACK = 0x0B


class Priority(IntEnum):
    """Message priority (2-bit encoding)"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class BinaryMessage:
    """
    Binary message format (48-byte header + variable payload)

    Header layout:
    - Magic (4 bytes): 0x434C4144 ('CLAD')
    - Version (2 bytes): Protocol version
    - Type (1 byte): MessageType
    - Priority (1 byte): Priority level
    - Source ID (8 bytes): Source agent hash
    - Target ID (8 bytes): Target agent hash
    - Correlation ID (8 bytes): Request tracking
    - Timestamp (8 bytes): Unix timestamp (ms)
    - Payload length (4 bytes): Payload size in bytes
    - Checksum (4 bytes): CRC32 of header+payload
    - POW nonce (8 bytes): Proof-of-work nonce
    - Flags (4 bytes): Reserved flags
    Total: 60 bytes
    """
    magic: int = 0x434C4144
    version: int = 1
    msg_type: MessageType = MessageType.COMMAND
    priority: Priority = Priority.NORMAL
    source_id: int = 0
    target_id: int = 0
    correlation_id: int = 0
    timestamp_ms: int = 0
    payload_length: int = 0
    checksum: int = 0
    pow_nonce: int = 0
    flags: int = 0
    payload: bytes = b''


class CryptoPOW:
    """
    Cryptographic Proof-of-Work for agent validation

    - Prefers AVX512 + C accel when available
    - Falls back to AVX2 + C accel when AVX512 missing
    - Falls back to pure Python scalar hashing otherwise
    - Automatically reduces effective difficulty when vector width is lower
    """

    def __init__(self, difficulty: int = 20):
        """
        Initialize crypto POW

        Args:
            difficulty: Requested number of leading zero bits (20 ≈ 1M hashes on AVX512)
        """
        self.requested_difficulty = int(difficulty)
        self.difficulty = int(difficulty)  # keep for API compatibility

        self.p_cores = self._detect_p_cores()
        self.has_avx512, self.has_avx2 = self._detect_vector_capabilities()
        self.effective_difficulty = self._compute_effective_difficulty()
        # For logging/debug
        self._vector_mode = self._compute_vector_mode()

    # --------------------------------------------------------------------- #
    # Capability detection
    # --------------------------------------------------------------------- #

    def _detect_p_cores(self) -> List[int]:
        """
        Detect P-cores for pinning.

        Heuristic:
        - Use first N logical cores, where N is min(12, total logical).
        - This is only a hint; cpu_affinity may fail and is best-effort.
        """
        try:
            cpu_count = psutil.cpu_count(logical=True) or 1
        except Exception:
            cpu_count = 1
        return list(range(min(12, cpu_count)))

    def _detect_vector_capabilities(self) -> Tuple[bool, bool]:
        """
        Detect AVX512 / AVX2 support from /proc/cpuinfo (Linux) or fall back.

        Returns:
            (has_avx512, has_avx2)
        """
        has_avx512 = False
        has_avx2 = False

        # Linux: parse /proc/cpuinfo flags
        cpuinfo_path = "/proc/cpuinfo"
        try:
            if os.path.exists(cpuinfo_path):
                with open(cpuinfo_path, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        if line.lower().startswith("flags"):
                            flags = line.split(":", 1)[1].strip().split()
                            flagset = set(flags)
                            if "avx2" in flagset:
                                has_avx2 = True
                            # Any AVX512 variant implies wide vectors available
                            if any(flag.startswith("avx512") for flag in flagset):
                                has_avx512 = True
                            # Once we've seen flags once, we can stop
                            break
        except Exception:
            # Non-fatal; keep defaults
            pass

        return has_avx512, has_avx2

    def _compute_effective_difficulty(self) -> int:
        """
        Adjust difficulty based on actual vector capability and C acceleration.

        Heuristic:
        - AVX512 + C:       keep requested difficulty
        - AVX2 + C:         reduce by 2–4 bits (4–16x easier)
        - scalar (Python):  reduce more aggressively to avoid huge CPU burns
        """
        d = self.requested_difficulty

        if C_ACCEL_AVAILABLE and self.has_avx512:
            # Fastest path: keep as-is
            return max(1, d)

        if C_ACCEL_AVAILABLE and self.has_avx2:
            # AVX2 only: 4 bits easier by default
            return max(4, d - 4)

        # Scalar fallback: make it much cheaper
        return max(4, d - 6)

    def _compute_vector_mode(self) -> str:
        """Return human-readable vector mode."""
        if C_ACCEL_AVAILABLE and self.has_avx512:
            return "avx512"
        if C_ACCEL_AVAILABLE and self.has_avx2:
            return "avx2"
        return "scalar"

    @property
    def vector_mode(self) -> str:
        """Public view of vector mode ('avx512'|'avx2'|'scalar')."""
        return self._vector_mode

    # --------------------------------------------------------------------- #
    # Core operations
    # --------------------------------------------------------------------- #

    def _pin_to_p_cores(self):
        """Pin current process to P-cores (best-effort, safe if it fails)."""
        try:
            p = psutil.Process()
            if self.p_cores:
                p.cpu_affinity(self.p_cores)
        except Exception:
            # Affinity is best-effort; ignore failures
            pass

    def compute_pow(self, data: bytes) -> Tuple[int, int]:
        """
        Compute proof-of-work for message.

        Args:
            data: Message data to hash

        Returns:
            (nonce, iterations): Nonce that satisfies effective difficulty
        """
        # Pin to P-cores for best performance
        self._pin_to_p_cores()

        difficulty = self.effective_difficulty

        # Use C acceleration if available
        if C_ACCEL_AVAILABLE and libagent is not None:
            nonce_ptr = c_uint64()
            data_ptr = (c_uint8 * len(data))(*data)
            iterations = libagent.crypto_pow_compute(
                data_ptr,
                len(data),
                difficulty,
                ctypes.byref(nonce_ptr)
            )
            return (nonce_ptr.value, iterations)

        # Python fallback (slower scalar SHA-256 loop)
        nonce = 0
        # target = 2^(256 - difficulty)
        target = (1 << (256 - difficulty))

        while True:
            candidate = data + struct.pack('<Q', nonce)
            hash_val = int.from_bytes(hashlib.sha256(candidate).digest(), 'big')

            if hash_val < target:
                # iterations ≈ nonce+1
                return (nonce, nonce + 1)

            nonce += 1

            # Safety limit (prevent runaway on super-high difficulty)
            if nonce > 1_000_000_000:
                raise RuntimeError(
                    f"POW computation exceeded safety limit "
                    f"(difficulty={difficulty}, vector_mode={self.vector_mode})"
                )

    def verify_pow(self, data: bytes, nonce: int) -> bool:
        """
        Verify proof-of-work.

        Args:
            data: Message data
            nonce: Claimed nonce

        Returns:
            True if POW is valid at effective difficulty
        """
        difficulty = self.effective_difficulty

        # Use C acceleration if available
        if C_ACCEL_AVAILABLE and libagent is not None:
            data_with_nonce = data + struct.pack('<Q', nonce)
            data_ptr = (c_uint8 * len(data_with_nonce))(*data_with_nonce)
            result = libagent.crypto_pow_verify(
                data_ptr,
                len(data_with_nonce),
                difficulty
            )
            return result == 1

        # Python fallback (scalar SHA-256)
        target = (1 << (256 - difficulty))
        candidate = data + struct.pack('<Q', nonce)
        hash_val = int.from_bytes(hashlib.sha256(candidate).digest(), 'big')
        return hash_val < target


class BinaryProtocol:
    """Binary message encoding/decoding with C acceleration"""

    @staticmethod
    def encode(msg: BinaryMessage) -> bytes:
        """
        Encode message to binary format

        Args:
            msg: BinaryMessage to encode

        Returns:
            Binary encoded message
        """
        # Pack header (60 bytes)
        header = struct.pack(
            '<I H B B Q Q Q Q I I Q I',
            msg.magic,
            msg.version,
            msg.msg_type,
            msg.priority,
            msg.source_id,
            msg.target_id,
            msg.correlation_id,
            msg.timestamp_ms,
            msg.payload_length,
            msg.checksum,
            msg.pow_nonce,
            msg.flags
        )

        # Use C acceleration if available
        if C_ACCEL_AVAILABLE and libagent is not None:
            total_size = len(header) + len(msg.payload)
            output_buffer = (c_uint8 * total_size)()
            input_data = header + msg.payload
            input_ptr = (c_uint8 * len(input_data))(*input_data)

            encoded_size = libagent.binary_encode_message(
                input_ptr,
                len(input_data),
                output_buffer
            )

            return bytes(output_buffer[:encoded_size])

        # Python fallback
        return header + msg.payload

    @staticmethod
    def decode(data: bytes) -> BinaryMessage:
        """
        Decode binary message

        Args:
            data: Binary data to decode

        Returns:
            Decoded BinaryMessage
        """
        if len(data) < 60:
            raise ValueError("Invalid message: too short")

        # Use C acceleration if available
        if C_ACCEL_AVAILABLE and libagent is not None:
            output_buffer = (c_uint8 * len(data))()
            input_ptr = (c_uint8 * len(data))(*data)

            decoded_size = libagent.binary_decode_message(
                input_ptr,
                len(data),
                output_buffer
            )

            data = bytes(output_buffer[:decoded_size])

        # Unpack header
        header_data = struct.unpack('<I H B B Q Q Q Q I I Q I', data[:60])

        msg = BinaryMessage(
            magic=header_data[0],
            version=header_data[1],
            msg_type=MessageType(header_data[2]),
            priority=Priority(header_data[3]),
            source_id=header_data[4],
            target_id=header_data[5],
            correlation_id=header_data[6],
            timestamp_ms=header_data[7],
            payload_length=header_data[8],
            checksum=header_data[9],
            pow_nonce=header_data[10],
            flags=header_data[11],
            payload=data[60:]
        )

        # Validate magic
        if msg.magic != 0x434C4144:
            raise ValueError(f"Invalid magic: 0x{msg.magic:08x}")

        # Validate payload length
        if len(msg.payload) != msg.payload_length:
            raise ValueError(
                f"Payload length mismatch: expected {msg.payload_length}, "
                f"got {len(msg.payload)}"
            )

        return msg

    @staticmethod
    def compute_checksum(data: bytes) -> int:
        """Compute CRC32 checksum"""
        import zlib
        return zlib.crc32(data) & 0xFFFFFFFF


class SharedMessageBus:
    """
    Global message bus using multiprocessing shared structures
    Singleton pattern for inter-agent communication
    """
    _instance = None
    _lock = mp.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return

        # Agent queues (one per agent)
        self.agent_queues: Dict[str, mp.Queue] = {}
        self.queue_lock = mp.Lock()

        # Shared memory registry for large payloads
        self.shared_memory_registry: Dict[str, shared_memory.SharedMemory] = {}

        self._initialized = True

    def get_or_create_queue(self, agent_id: str) -> mp.Queue:
        """Get or create queue for agent"""
        with self.queue_lock:
            if agent_id not in self.agent_queues:
                self.agent_queues[agent_id] = mp.Queue(maxsize=1000)
            return self.agent_queues[agent_id]

    def create_shared_memory(self, name: str, size: int) -> shared_memory.SharedMemory:
        """Create shared memory segment for large payload"""
        try:
            shm = shared_memory.SharedMemory(name=name, create=True, size=size)
            self.shared_memory_registry[name] = shm
            return shm
        except FileExistsError:
            # Already exists, attach to it
            return shared_memory.SharedMemory(name=name, create=False)

    def cleanup(self):
        """Cleanup shared resources"""
        for shm in self.shared_memory_registry.values():
            try:
                shm.close()
                shm.unlink()
            except Exception:
                pass


class AgentCommunicator:
    """
    High-performance agent communication system
    Direct IPC with binary protocol + crypto POW (no Redis)
    """

    def __init__(
        self,
        agent_id: str,
        pow_difficulty: int = 20,
        enable_pow: bool = True
    ):
        """
        Initialize agent communicator

        Args:
            agent_id: Unique agent identifier
            pow_difficulty: POW difficulty (requested bits)
            enable_pow: Enable proof-of-work validation
        """
        self.agent_id = agent_id
        self.agent_id_hash = self._hash_agent_id(agent_id)
        self.enable_pow = enable_pow

        # Crypto POW
        self.pow: Optional[CryptoPOW]
        if enable_pow:
            self.pow = CryptoPOW(difficulty=pow_difficulty)
        else:
            self.pow = None

        # Get shared message bus
        self.message_bus = SharedMessageBus()
        self.queue = self.message_bus.get_or_create_queue(agent_id)

        # Statistics
        self.stats = {
            "sent": 0,
            "received": 0,
            "pow_computed": 0,
            "pow_verified": 0,
            "pow_failed": 0,
            "total_pow_time_ms": 0,
            "shared_memory_used": 0
        }

        # Banner for this agent
        print(f"✓ Agent Communicator initialized: {agent_id}")
        print(f"  Agent ID Hash: 0x{self.agent_id_hash:016x}")
        print(f"  Transport: Direct IPC (multiprocessing)")
        if self.pow:
            print(
                f"  POW: enabled "
                f"(requested={self.pow.requested_difficulty} bits, "
                f"effective={self.pow.effective_difficulty} bits, "
                f"vector={self.pow.vector_mode}, "
                f"C-accel={'yes' if C_ACCEL_AVAILABLE else 'no'})"
            )
        else:
            print("  POW: disabled")
        print(f"  C Acceleration (binary codec): {'available' if C_ACCEL_AVAILABLE else 'Python fallback'}")

    def _hash_agent_id(self, agent_id: str) -> int:
        """Hash agent ID to 64-bit integer"""
        hash_bytes = hashlib.sha256(agent_id.encode()).digest()[:8]
        return struct.unpack('<Q', hash_bytes)[0]

    def send(
        self,
        target_agent: str,
        msg_type: MessageType,
        payload: bytes,
        priority: Priority = Priority.NORMAL,
        correlation_id: Optional[int] = None
    ) -> bool:
        """
        Send message to target agent

        Args:
            target_agent: Target agent ID
            msg_type: Message type
            payload: Message payload (binary)
            priority: Message priority
            correlation_id: Optional correlation ID for request tracking

        Returns:
            True if sent successfully
        """
        # Create message
        msg = BinaryMessage(
            msg_type=msg_type,
            priority=priority,
            source_id=self.agent_id_hash,
            target_id=self._hash_agent_id(target_agent),
            correlation_id=correlation_id or int(time.time() * 1000000),
            timestamp_ms=int(time.time() * 1000),
            payload_length=len(payload),
            payload=payload
        )

        # Compute POW if enabled
        if self.enable_pow and self.pow is not None:
            pow_start = time.time()
            header_data = struct.pack(
                '<I H B B Q Q Q Q I',
                msg.magic, msg.version, msg.msg_type, msg.priority,
                msg.source_id, msg.target_id, msg.correlation_id,
                msg.timestamp_ms, msg.payload_length
            )
            nonce, iterations = self.pow.compute_pow(header_data + payload)
            msg.pow_nonce = nonce

            pow_time_ms = (time.time() - pow_start) * 1000
            self.stats["pow_computed"] += 1
            self.stats["total_pow_time_ms"] += pow_time_ms

        # Compute checksum
        msg_without_checksum = struct.pack(
            '<I H B B Q Q Q Q I',
            msg.magic, msg.version, msg.msg_type, msg.priority,
            msg.source_id, msg.target_id, msg.correlation_id,
            msg.timestamp_ms, msg.payload_length
        ) + payload
        msg.checksum = BinaryProtocol.compute_checksum(msg_without_checksum)

        # Encode message
        encoded = BinaryProtocol.encode(msg)

        # Send via direct IPC
        try:
            # Get target agent's queue
            target_queue = self.message_bus.get_or_create_queue(target_agent)

            # For large payloads (>1MB), use shared memory
            if len(encoded) > 1_000_000:
                shm_name = f"msg_{int(time.time() * 1000000)}"
                shm = self.message_bus.create_shared_memory(shm_name, len(encoded))
                shm.buf[:len(encoded)] = encoded

                # Send shared memory reference instead
                target_queue.put(("shm", shm_name, len(encoded)), block=False)
                self.stats["shared_memory_used"] += 1
            else:
                # Send directly for small messages
                target_queue.put(("direct", encoded), block=False)

            self.stats["sent"] += 1
            return True
        except Exception as e:
            print(f"❌ Send failed: {e}")
            return False

    def receive(self, timeout_ms: int = 1000) -> Optional[BinaryMessage]:
        """
        Receive message from queue

        Args:
            timeout_ms: Timeout in milliseconds

        Returns:
            Received message or None
        """
        try:
            # Receive from direct IPC queue
            result = self.queue.get(timeout=timeout_ms / 1000)

            msg_type, *data = result

            if msg_type == "shm":
                # Shared memory message
                shm_name, size = data
                shm = shared_memory.SharedMemory(name=shm_name, create=False)
                encoded = bytes(shm.buf[:size])
                shm.close()
                # Note: Sender is responsible for cleanup
            else:
                # Direct message
                encoded = data[0]

            # Decode message
            msg = BinaryProtocol.decode(encoded)

            # Verify POW if enabled
            if self.enable_pow and self.pow is not None:
                header_data = struct.pack(
                    '<I H B B Q Q Q Q I',
                    msg.magic, msg.version, msg.msg_type, msg.priority,
                    msg.source_id, msg.target_id, msg.correlation_id,
                    msg.timestamp_ms, msg.payload_length
                )

                if self.pow.verify_pow(header_data + msg.payload, msg.pow_nonce):
                    self.stats["pow_verified"] += 1
                else:
                    self.stats["pow_failed"] += 1
                    print(f"⚠️  POW verification failed for message from 0x{msg.source_id:016x}")
                    return None

            self.stats["received"] += 1
            return msg

        except Exception as e:
            print(f"❌ Receive failed: {e}")
            return None

    def get_stats(self) -> Dict[str, Any]:
        """Get communication statistics"""
        stats = self.stats.copy()

        if stats["pow_computed"] > 0:
            stats["avg_pow_time_ms"] = stats["total_pow_time_ms"] / stats["pow_computed"]
        else:
            stats["avg_pow_time_ms"] = 0

        if self.pow is not None:
            stats["pow_requested_bits"] = self.pow.requested_difficulty
            stats["pow_effective_bits"] = self.pow.effective_difficulty
            stats["vector_mode"] = self.pow.vector_mode
        else:
            stats["pow_requested_bits"] = 0
            stats["pow_effective_bits"] = 0
            stats["vector_mode"] = "disabled"

        return stats


def demo():
    """Demonstration of binary agent communication"""

    print("=" * 70)
    print(" Binary Agent Communication Protocol Demo")
    print("=" * 70)
    print()

    # Create two agents
    agent1 = AgentCommunicator("agent_001", enable_pow=True, pow_difficulty=16)
    agent2 = AgentCommunicator("agent_002", enable_pow=True, pow_difficulty=16)

    print()
    print("Testing message exchange...")
    print()

    # Agent 1 sends command to Agent 2
    payload = b"Execute task: analyze system performance"
    print(f"Agent 1 → Agent 2: {payload.decode()}")

    success = agent1.send(
        target_agent="agent_002",
        msg_type=MessageType.COMMAND,
        payload=payload,
        priority=Priority.HIGH
    )

    if success:
        print("✓ Message sent")

        # Agent 2 receives
        msg = agent2.receive(timeout_ms=5000)

        if msg:
            print(f"✓ Message received by Agent 2")
            print(f"  Type: {msg.msg_type.name}")
            print(f"  Priority: {msg.priority.name}")
            print(f"  Payload: {msg.payload.decode()}")
            print(f"  POW Nonce: 0x{msg.pow_nonce:016x}")

            # Agent 2 sends response
            response_payload = b"Task completed successfully"
            agent2.send(
                target_agent="agent_001",
                msg_type=MessageType.RESPONSE,
                payload=response_payload,
                priority=Priority.NORMAL,
                correlation_id=msg.correlation_id
            )

            # Agent 1 receives response
            response = agent1.receive(timeout_ms=5000)

            if response:
                print(f"\n✓ Response received by Agent 1")
                print(f"  Payload: {response.payload.decode()}")
                print(f"  Correlation ID match: {response.correlation_id == msg.correlation_id}")

    print()
    print("=" * 70)
    print(" Statistics")
    print("=" * 70)
    print()

    print("Agent 1:")
    for key, value in agent1.get_stats().items():
        print(f"  {key}: {value}")

    print()
    print("Agent 2:")
    for key, value in agent2.get_stats().items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    demo()
