# Fast Date Algorithm Implementation Plan

## Overview
Implement the Joffe "Century-February-Padding" algorithm from https://www.benjoffe.com/fast-date in LLVM libc's time utilities to achieve 2-11% performance improvement over the current implementation.

## Current State Analysis

### Existing Implementation (`time_utils.cpp`)
- **Function**: `update_from_seconds()` - Converts time_t to year/month/day
- **Approach**: Traditional slicing method
  - Divides timeline into 400-year cycles
  - Then 100-year cycles  
  - Then 4-year cycles
  - Finally individual years
- **Performance**: Uses multiple divisions and multiplications by large constants

### Target for Optimization
The `update_from_seconds()` function that converts a time_t timestamp into a `struct tm` with year, month, day components.

## Proposed Implementation

### Phase 1: Core Algorithm Implementation
**File**: `libc/src/time/fast_date.h` (new)
- Implement Joffe's fast year calculation algorithm
- Key innovation: Map Gregorian calendar to Julian by padding with fake Feb 29s
- Algorithm steps:
  ```cpp
  // Convert days since epoch to year
  days += EPOCH_SHIFT + 306          // Shift epoch and start from March
  qday = days * 4 + 3                // Quarter-days since 0000-02-28 06:00
  cent = qday / 146097               // Century-Februaries elapsed
  qjul = qday - (cent & ~3) + cent * 4  // Map to Julian Quarter-Day
  year = qjul / 1461                 // Year (incremented later if Jan/Feb)
  yday = (qjul % 1461) / 4           // Day of Year (starting 1 March)
  ```

**File**: `libc/src/time/fast_date.cpp` (new)
- Implement complete fast date conversion
- Use Neri-Schneider EAF for month/day calculation:
  ```cpp
  N = yday * 2141 + 197913
  M = N / 65536
  D = N % 65536 / 2141
  ```
- Handle January/February bump correctly
- Maintain compatibility with existing `struct tm` format

### Phase 2: Integration Points

**Option A: Replace existing algorithm**
- Modify `update_from_seconds()` in `time_utils.cpp`
- Direct drop-in replacement
- Risk: May break existing code if edge cases differ

**Option B: Add parallel implementation**
- Create `update_from_seconds_fast()` alongside existing
- Allows A/B comparison
- Can be feature-flagged
- Recommended for initial implementation

### Phase 3: Inverse Function Optimization
**File**: `mktime_internal()` in `time_utils.cpp`
- Current approach: Calculates days from year/month/day
- Optimization from article:
  - Change `year * 1461 / 4` to `year * 365 + year / 4`
  - Avoids overflow, covers full 32/64-bit range
  - ~4% faster

### Phase 4: Testing & Validation

**File**: `libc/test/src/time/fast_date_test.cpp` (new)
Test coverage:
- [ ] Correctness: Compare against existing implementation
  - Test all dates from 1900-2100
  - Edge cases: leap years (1900, 2000, 2004, 2100)
  - Boundary dates: epoch, 32-bit limits, 64-bit limits
- [ ] Compatibility: Ensure identical output to current implementation
  - Same `tm` structure values
  - Same error handling for out-of-range dates
- [ ] Performance regression tests

**File**: `libc/benchmarks/src/time/date_conversion_benchmark.cpp` (new)
Benchmark suite:
- [ ] Sequential date conversion (measure cache effects)
- [ ] Random date conversion (real-world usage)
- [ ] Year-only extraction (isolated optimization)
- [ ] Full date conversion (year + month + day)
- [ ] Inverse function (mktime)
- [ ] Compare: old vs new vs system libc

### Phase 5: Documentation

**Update**: `libc/src/time/time_utils.h`
- Add comments explaining the algorithm
- Document the Century-February-Padding technique
- Credit Ben Joffe's work with link

**Create**: `libc/src/time/FAST_DATE_ALGORITHM.md` (new)
- Detailed explanation of the optimization
- Performance characteristics
- Overflow behavior
- Comparison with Neri-Schneider

## Expected Performance Gains

Based on article benchmarks across different architectures:
- **ARM (Snapdragon)**: 8.7% faster
- **x86 (Intel i3)**: >9.3% faster  
- **Apple M4 Pro**: 4.4% faster
- **Intel Core i5**: 2.5% faster

Target for LLVM libc: **5-10% improvement** in date conversion performance.

## Trade-offs & Considerations

### Advantages
✅ Simpler algorithm (fewer operations)
✅ 2-11% faster across platforms
✅ Easier to understand (no nested slicing)
✅ Inverse function can avoid overflow completely

### Disadvantages
⚠️ Overflow 0.002% earlier (3 days per 400 years padding)
⚠️ Different intermediate values (may affect debugging)
⚠️ Need to validate correctness thoroughly

### Compatibility
- Must maintain exact same `struct tm` output
- Must handle same date ranges (or document differences)
- Error handling must be identical

## Implementation Order

1. **Create fast_date.h/.cpp** with new algorithm
2. **Add comprehensive unit tests** to verify correctness
3. **Create benchmarks** to measure actual speedup
4. **Integrate into time_utils.cpp** as opt-in variant
5. **Run full test suite** to catch regressions
6. **Benchmark on multiple architectures** in dev container
7. **Document results** and make recommendation
8. **If successful**: Make default, gate behind feature flag
9. **Update mktime** with overflow-safe inverse function

## Success Criteria

- ✅ All existing tests pass
- ✅ New implementation produces identical results to old
- ✅ Performance improvement of 5%+ on at least 2 architectures
- ✅ No increase in binary size >1KB
- ✅ Code review approval from LLVM libc maintainers
- ✅ Full documentation and comments

## Timeline Estimate

- Phase 1 (Implementation): 2-4 hours
- Phase 2 (Integration): 1-2 hours  
- Phase 3 (Inverse optimization): 1-2 hours
- Phase 4 (Testing): 3-5 hours
- Phase 5 (Documentation): 1-2 hours
- **Total**: 8-15 hours

## References

- Original article: https://www.benjoffe.com/fast-date
- Neri-Schneider paper: https://onlinelibrary.wiley.com/doi/full/10.1002/spe.3172
- Howard Hinnant date algorithms: https://howardhinnant.github.io/date_algorithms.html
- LLVM libc time implementation: `libc/src/time/time_utils.cpp`

## Next Steps

1. Review this plan with team
2. Get approval for approach (Option A vs B)
3. Start with Phase 1: Core algorithm implementation
4. Create feature branch: `feature/fast-date-algorithm`
