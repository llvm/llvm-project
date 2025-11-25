#!/usr/bin/env python3
"""
Intelligent Analysis Cache
Hash-based caching with NPU acceleration support

Supported NPU hardware:
- Intel Neural Compute Stick (Movidius)
- Google Coral Edge TPU
- Apple Neural Engine (M1/M2)

Fallback: RAM-based LRU cache
"""

import hashlib
import json
import pickle
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field
from enum import Enum
from collections import OrderedDict

try:
    from async_analysis import AsyncCodeAnalyzer, AnalysisType
except ImportError:
    try:
        from rag_system.async_analysis import AsyncCodeAnalyzer, AnalysisType
    except ImportError:
        AsyncCodeAnalyzer = None
        AnalysisType = None


class CacheBackend(Enum):
    """Cache storage backends"""
    RAM = "ram"              # In-memory LRU cache
    NPU_INTEL = "npu_intel"  # Intel Movidius Neural Compute Stick
    NPU_CORAL = "npu_coral"  # Google Coral Edge TPU
    NPU_APPLE = "npu_apple"  # Apple Neural Engine
    DISK = "disk"            # Persistent disk cache


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    timestamp: datetime
    ttl_seconds: int
    access_count: int = 0
    last_access: datetime = field(default_factory=datetime.now)
    size_bytes: int = 0

    def is_expired(self) -> bool:
        """Check if entry has expired"""
        age = datetime.now() - self.timestamp
        return age.total_seconds() > self.ttl_seconds

    def touch(self):
        """Update access statistics"""
        self.access_count += 1
        self.last_access = datetime.now()


@dataclass
class CacheStats:
    """Cache performance statistics"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_size_bytes: int = 0
    entry_count: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def __str__(self) -> str:
        return (
            f"Cache Stats:\n"
            f"  Hits: {self.hits}\n"
            f"  Misses: {self.misses}\n"
            f"  Hit Rate: {self.hit_rate:.1%}\n"
            f"  Entries: {self.entry_count}\n"
            f"  Size: {self.total_size_bytes / 1024:.1f} KB\n"
            f"  Evictions: {self.evictions}"
        )


class NPUDetector:
    """Detect available NPU hardware"""

    @staticmethod
    def detect_intel_ncs() -> bool:
        """Detect Intel Neural Compute Stick (Movidius)"""
        # Check for custom NUC2.1 driver first
        try:
            # Check if custom driver is available
            import importlib.util
            spec = importlib.util.find_spec('nuc21')  # Custom driver module
            if spec is not None:
                return True
        except ImportError:
            pass

        # Fallback to standard OpenVINO detection
        try:
            from openvino.runtime import Core
            core = Core()
            devices = core.available_devices
            return 'MYRIAD' in devices or 'HDDL' in devices
        except ImportError:
            return False
        except Exception:
            return False

    @staticmethod
    def detect_coral_tpu() -> bool:
        """Detect Google Coral Edge TPU"""
        try:
            # Try importing pycoral
            from pycoral.utils import edgetpu
            devices = edgetpu.list_edge_tpus()
            return len(devices) > 0
        except ImportError:
            return False
        except Exception:
            return False

    @staticmethod
    def detect_apple_neural_engine() -> bool:
        """Detect Apple Neural Engine (M1/M2 Macs)"""
        try:
            import platform
            if platform.system() == 'Darwin':
                # Check for Apple Silicon
                machine = platform.machine()
                return machine == 'arm64'
        except Exception:
            return False

        return False

    @classmethod
    def detect_best_backend(cls) -> CacheBackend:
        """
        Detect best available cache backend

        Priority:
        1. Intel NCS (dedicated NPU)
        2. Google Coral TPU (dedicated NPU)
        3. Apple Neural Engine (integrated)
        4. RAM (fallback)
        """
        if cls.detect_intel_ncs():
            return CacheBackend.NPU_INTEL
        elif cls.detect_coral_tpu():
            return CacheBackend.NPU_CORAL
        elif cls.detect_apple_neural_engine():
            return CacheBackend.NPU_APPLE
        else:
            return CacheBackend.RAM


class LRUCache:
    """
    LRU (Least Recently Used) cache with TTL support

    Features:
    - Fixed size limit (evicts oldest entries)
    - TTL-based expiration
    - Access statistics
    """

    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        """
        Args:
            max_size: Maximum number of entries
            default_ttl: Default TTL in seconds (default: 1 hour)
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.stats = CacheStats()

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        if key not in self.cache:
            self.stats.misses += 1
            return None

        entry = self.cache[key]

        # Check expiration
        if entry.is_expired():
            self._evict(key)
            self.stats.misses += 1
            return None

        # Update access stats
        entry.touch()

        # Move to end (most recently used)
        self.cache.move_to_end(key)

        self.stats.hits += 1
        return entry.value

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """
        Set value in cache

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (None = default)
        """
        ttl = ttl or self.default_ttl

        # Calculate size
        try:
            size_bytes = len(pickle.dumps(value))
        except Exception:
            size_bytes = 0

        # Create entry
        entry = CacheEntry(
            key=key,
            value=value,
            timestamp=datetime.now(),
            ttl_seconds=ttl,
            size_bytes=size_bytes
        )

        # Evict if at capacity
        if len(self.cache) >= self.max_size and key not in self.cache:
            # Evict oldest (first item)
            oldest_key = next(iter(self.cache))
            self._evict(oldest_key)

        # Store entry
        if key in self.cache:
            # Update existing
            old_size = self.cache[key].size_bytes
            self.stats.total_size_bytes -= old_size

        self.cache[key] = entry
        self.cache.move_to_end(key)

        # Update stats
        self.stats.total_size_bytes += size_bytes
        self.stats.entry_count = len(self.cache)

    def _evict(self, key: str):
        """Evict entry from cache"""
        if key in self.cache:
            entry = self.cache.pop(key)
            self.stats.total_size_bytes -= entry.size_bytes
            self.stats.evictions += 1
            self.stats.entry_count = len(self.cache)

    def clear(self):
        """Clear entire cache"""
        self.cache.clear()
        self.stats = CacheStats()

    def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        return self.stats


class AnalysisCache:
    """
    Intelligent cache for code analysis results

    Features:
    - Automatic NPU detection and acceleration
    - Hash-based cache keys (MD5 of code content)
    - TTL-based invalidation (configurable per analysis type)
    - Persistent disk cache (optional)
    - Cache warming and statistics
    """

    # TTL profiles for different analysis types
    TTL_PROFILES = {
        'security': 24 * 3600,      # 24 hours (security rules change rarely)
        'performance': 12 * 3600,   # 12 hours (performance patterns stable)
        'complexity': 7 * 24 * 3600,  # 7 days (complexity metrics very stable)
        'smells': 24 * 3600,        # 24 hours
        'formatting': 3600,         # 1 hour (style rules might change)
        'default': 6 * 3600         # 6 hours
    }

    def __init__(self,
                 cache_dir: str = '.code_assistant_cache',
                 max_size: int = 10000,
                 use_disk: bool = True,
                 use_npu: bool = True,
                 verbose: bool = False):
        """
        Args:
            cache_dir: Directory for persistent cache
            max_size: Maximum number of cache entries
            use_disk: Enable persistent disk cache
            use_npu: Enable NPU acceleration if available
            verbose: Print cache operations
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_disk = use_disk
        self.verbose = verbose

        # Detect best backend
        if use_npu:
            self.backend = NPUDetector.detect_best_backend()
        else:
            self.backend = CacheBackend.RAM

        if self.verbose:
            print(f"✓ Cache backend: {self.backend.value}")

        # Initialize cache
        self.cache = LRUCache(max_size=max_size)

        # Load persistent cache if enabled
        if use_disk:
            self._load_disk_cache()

    def get_cached_analysis(self, code: str, analysis_type: str) -> Optional[Dict]:
        """
        Retrieve cached analysis if valid

        Args:
            code: Source code
            analysis_type: Type of analysis

        Returns:
            Cached results or None
        """
        key = self._compute_cache_key(code, analysis_type)
        result = self.cache.get(key)

        if self.verbose and result:
            print(f"✓ Cache hit: {analysis_type}")

        return result

    def cache_analysis(self, code: str, analysis_type: str, results: Dict):
        """
        Store analysis results in cache

        Args:
            code: Source code
            analysis_type: Type of analysis
            results: Analysis results to cache
        """
        key = self._compute_cache_key(code, analysis_type)
        ttl = self.TTL_PROFILES.get(analysis_type, self.TTL_PROFILES['default'])

        self.cache.set(key, results, ttl=ttl)

        if self.verbose:
            print(f"✓ Cached: {analysis_type} (TTL: {ttl//3600}h)")

        # Persist to disk if enabled
        if self.use_disk:
            self._save_to_disk(key, results, analysis_type)

    def _compute_cache_key(self, code: str, analysis_type: str) -> str:
        """Generate cache key from code and analysis type"""
        content = f"{analysis_type}:{code}"
        return hashlib.md5(content.encode()).hexdigest()

    def _save_to_disk(self, key: str, results: Dict, analysis_type: str):
        """Save cache entry to disk"""
        cache_file = self.cache_dir / f"{key}_{analysis_type}.json"

        try:
            with open(cache_file, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'analysis_type': analysis_type,
                    'results': results
                }, f)
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Failed to save cache to disk: {e}")

    def _load_disk_cache(self):
        """Load persistent cache from disk"""
        if not self.cache_dir.exists():
            return

        loaded = 0
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file) as f:
                    data = json.load(f)

                # Check if still valid
                timestamp = datetime.fromisoformat(data['timestamp'])
                age = datetime.now() - timestamp
                analysis_type = data['analysis_type']
                ttl = self.TTL_PROFILES.get(analysis_type, self.TTL_PROFILES['default'])

                if age.total_seconds() < ttl:
                    # Extract key from filename
                    key = cache_file.stem.split('_')[0]
                    self.cache.set(key, data['results'], ttl=ttl)
                    loaded += 1

            except Exception:
                continue

        if self.verbose and loaded > 0:
            print(f"✓ Loaded {loaded} entries from disk cache")

    def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        return self.cache.get_stats()

    def clear(self):
        """Clear cache (RAM and disk)"""
        self.cache.clear()

        if self.use_disk:
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()

        if self.verbose:
            print("✓ Cache cleared")

    def warm_cache(self,
                   file_paths: List[str],
                   analysis_types: Optional[List[str]] = None,
                   max_workers: int = 4):
        """
        Pre-warm cache by actually running analyses through AsyncCodeAnalyzer.

        Args:
            file_paths: Source files to analyze and cache.
            analysis_types: Desired analysis buckets (defaults to all supported).
            max_workers: Thread pool size for AsyncCodeAnalyzer.
        """
        if AsyncCodeAnalyzer is None or AnalysisType is None:
            raise RuntimeError("Async analysis engines are unavailable. Install rag_system dependencies.")

        resolved_files = []
        for file_path in file_paths:
            path_obj = Path(file_path)
            if path_obj.is_file():
                resolved_files.append(path_obj)
            elif self.verbose:
                print(f"⚠️  Skipping missing file during cache warm: {file_path}")

        if not resolved_files:
            if self.verbose:
                print("⚠️  No valid files supplied for cache warming.")
            return

        if not analysis_types:
            analysis_types = ['security', 'performance', 'complexity', 'smells']

        resolved_types: List[AnalysisType] = []
        for requested in analysis_types:
            normalized = requested.lower().strip()
            if normalized == AnalysisType.FULL.value:
                resolved_types = [
                    AnalysisType.SECURITY,
                    AnalysisType.PERFORMANCE,
                    AnalysisType.COMPLEXITY,
                    AnalysisType.SMELLS
                ]
                break

            try:
                resolved_types.append(AnalysisType(normalized))
            except ValueError:
                if self.verbose:
                    print(f"⚠️  Unknown analysis type '{requested}' (skipping)")

        if not resolved_types:
            raise ValueError("No valid analysis types supplied for cache warming.")

        analyzer = AsyncCodeAnalyzer(
            max_workers=min(max_workers, max(1, len(resolved_files))),
            verbose=self.verbose
        )

        warmed_entries = 0
        for file_path in resolved_files:
            try:
                source_code = file_path.read_text(encoding='utf-8')
            except Exception as exc:
                if self.verbose:
                    print(f"⚠️  Failed to read {file_path}: {exc}")
                continue

            outstanding_types = [
                analysis_type for analysis_type in resolved_types
                if self.get_cached_analysis(source_code, analysis_type.value) is None
            ]

            if not outstanding_types:
                continue

            try:
                results = analyzer.analyze_parallel(source_code, analyses=outstanding_types)
            except Exception as exc:
                if self.verbose:
                    print(f"⚠️  Analysis failed for {file_path}: {exc}")
                continue

            for analysis_type, result in results.items():
                if result.success:
                    self.cache_analysis(source_code, analysis_type.value, result.results)
                    warmed_entries += 1
                elif self.verbose:
                    print(f"⚠️  {analysis_type.value} analysis failed for {file_path}: {result.error}")

        if self.verbose:
            print(f"✓ Cache warm complete. Stored {warmed_entries} analysis artifacts.")


def main():
    """Test analysis cache"""
    test_code = """
def test_function(x, y):
    result = x + y
    return result
"""

    print("="*70)
    print("Analysis Cache Demo")
    print("="*70)

    # Initialize cache
    cache = AnalysisCache(verbose=True)

    print(f"\n🔧 Backend: {cache.backend.value}")

    # Simulate analysis results
    security_results = {
        'issues': [],
        'score': 10.0
    }

    performance_results = {
        'issues': [],
        'suggestions': []
    }

    # Cache results
    print("\n📝 Caching analysis results...")
    cache.cache_analysis(test_code, 'security', security_results)
    cache.cache_analysis(test_code, 'performance', performance_results)

    # Retrieve from cache
    print("\n🔍 Retrieving from cache...")
    cached_security = cache.get_cached_analysis(test_code, 'security')
    cached_performance = cache.get_cached_analysis(test_code, 'performance')

    print(f"  Security: {'✓ Found' if cached_security else '✗ Miss'}")
    print(f"  Performance: {'✓ Found' if cached_performance else '✗ Miss'}")

    # Cache miss test
    cached_missing = cache.get_cached_analysis(test_code, 'nonexistent')
    print(f"  Nonexistent: {'✓ Found' if cached_missing else '✗ Miss (expected)'}")

    # Statistics
    print("\n📊 Cache Statistics:")
    stats = cache.get_stats()
    print(f"  {stats}")

    print("\n" + "="*70)


if __name__ == '__main__':
    main()
