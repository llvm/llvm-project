//===-- precision_analysis_runtime.cpp - Precision Analysis Runtime ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements runtime for analyzing whether floating-point operations
// could be performed with lower precision while maintaining acceptable
// accuracy. It instruments FP operations, simulates them with lower precision,
// and compares results to determine if precision reduction is viable.
//
//===----------------------------------------------------------------------===//

#include "../instrumentor_runtime.h"

#include <atomic>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>

// Configuration: relative error threshold for "acceptable" lower precision
// A result is considered acceptable if |result_lower - result_higher| /
// |result_higher| < threshold
static constexpr double DefaultRelativeErrorThreshold = 1e-3; // 0.1%

// Per-operation statistics - tracks separately by original precision
struct OperationStats {
  uint64_t TotalCount; // Total number of times this operation executed

  // Double-precision operations (started as double)
  uint64_t DoubleToFp16;      // Double ops that can use fp16
  uint64_t DoubleToFloat;     // Double ops that can use float (but not fp16)
  uint64_t DoubleNeedsDouble; // Double ops that need double precision

  // Float-precision operations (started as float)
  uint64_t FloatToFp16;     // Float ops that can use fp16
  uint64_t FloatNeedsFloat; // Float ops that need float precision

  // Special values
  uint64_t
      InputSpecialValues; // Times when inputs had special values (NaN, Inf)
  uint64_t DoubleLoweringSpecial; // Double ops where lowering caused overflow
  uint64_t FloatLoweringSpecial;  // Float ops where lowering caused overflow
};

// Helper functions to get statistics map and mutex
// Using function-local statics ensures proper initialization order
// and avoids static destruction order fiasco.
//
// IMPORTANT: We use heap allocation (new) without delete to intentionally
// "leak" these objects. This ensures they remain valid when the destructor
// function runs at program exit, even if it runs after static destructors.
// For a profiling tool that runs once and exits, this is acceptable.
static std::map<int32_t, OperationStats> &getOperationStats() {
  static std::map<int32_t, OperationStats> *Stats =
      new std::map<int32_t, OperationStats>();
  return *Stats;
}

static std::mutex &getStatsMutex() {
  static std::mutex *Mutex = new std::mutex();
  return *Mutex;
}

enum {
  LLVM_OPCODE_FAdd = 15,
  LLVM_OPCODE_FSub = 17,
  LLVM_OPCODE_FMul = 19,
  LLVM_OPCODE_FDiv = 22,
  LLVM_OPCODE_FRem = 25,
  LLVM_OPCODE_FNeg = 13,
};

// Helper: Convert float to fp16 (IEEE 754 half precision) and back
// fp16 format: 1 sign bit, 5 exponent bits, 10 mantissa bits
static inline float simulateFp16Precision(float Value) {
  // Handle special cases
  if (std::isnan(Value) || std::isinf(Value)) {
    return Value;
  }

  uint32_t Bits;
  std::memcpy(&Bits, &Value, sizeof(float));

  uint32_t Sign = Bits & 0x80000000u;
  int32_t Exponent = ((Bits >> 23) & 0xFF) - 127;
  uint32_t Mantissa = Bits & 0x7FFFFFu;

  // fp16 range: exponent -14 to +15 (biased 1 to 30)
  // Underflow to zero
  if (Exponent < -14) {
    return Sign ? -0.0f : 0.0f;
  }

  // Overflow to infinity
  if (Exponent > 15) {
    return Sign ? -INFINITY : INFINITY;
  }

  // Round mantissa from 23 bits to 10 bits
  uint32_t Fp16Mantissa = (Mantissa + 0x1000u) >> 13;
  if (Fp16Mantissa > 0x3FF) {
    // Rounding caused overflow
    Fp16Mantissa = 0;
    Exponent++;
    if (Exponent > 15) {
      return Sign ? -INFINITY : INFINITY;
    }
  }

  // Reconstruct float with reduced precision
  uint32_t Fp16Exponent = (Exponent + 127) & 0xFF;
  uint32_t ResultBits = Sign | (Fp16Exponent << 23) | (Fp16Mantissa << 13);

  float Result;
  std::memcpy(&Result, &ResultBits, sizeof(float));
  return Result;
}

// Helper: Check if value is special (NaN or Inf)
static inline bool isSpecialValue(double Value) {
  return std::isnan(Value) || std::isinf(Value);
}

static inline bool isSpecialValue(float Value) {
  return std::isnan(Value) || std::isinf(Value);
}

// Helper: Compute relative error
static inline double computeRelativeError(double Reference, double Test) {
  if (Reference == 0.0) {
    return (Test == 0.0) ? 0.0 : INFINITY;
  }
  return std::fabs((Test - Reference) / Reference);
}

// Helper: Perform operation with lower precision (double → float)
static double simulateLowerPrecisionOp(int32_t Opcode, double Left,
                                       double Right) {
  float LeftF = static_cast<float>(Left);
  float RightF = static_cast<float>(Right);
  float ResultF = 0.0f;

  switch (Opcode) {
  case LLVM_OPCODE_FAdd:
    ResultF = LeftF + RightF;
    break;
  case LLVM_OPCODE_FSub:
    ResultF = LeftF - RightF;
    break;
  case LLVM_OPCODE_FMul:
    ResultF = LeftF * RightF;
    break;
  case LLVM_OPCODE_FDiv:
    ResultF = LeftF / RightF;
    break;
  case LLVM_OPCODE_FRem:
    ResultF = std::fmod(LeftF, RightF);
    break;
  case LLVM_OPCODE_FNeg:
    ResultF = -LeftF;
    break;
  default:
    // For unknown operations, assume lower precision is not ok
    return NAN;
  }

  return static_cast<double>(ResultF);
}

// Helper: Perform operation with fp16 precision (float → fp16)
static float simulateFp16Op(int32_t Opcode, float Left, float Right) {
  float LeftFp16 = simulateFp16Precision(Left);
  float RightFp16 = simulateFp16Precision(Right);
  float ResultFp16 = 0.0f;

  switch (Opcode) {
  case LLVM_OPCODE_FAdd:
    ResultFp16 = LeftFp16 + RightFp16;
    break;
  case LLVM_OPCODE_FSub:
    ResultFp16 = LeftFp16 - RightFp16;
    break;
  case LLVM_OPCODE_FMul:
    ResultFp16 = LeftFp16 * RightFp16;
    break;
  case LLVM_OPCODE_FDiv:
    ResultFp16 = LeftFp16 / RightFp16;
    break;
  case LLVM_OPCODE_FRem:
    ResultFp16 = std::fmod(LeftFp16, RightFp16);
    break;
  case LLVM_OPCODE_FNeg:
    ResultFp16 = -LeftFp16;
    break;
  default:
    return NAN;
  }

  // Apply fp16 precision to result as well
  return simulateFp16Precision(ResultFp16);
}

// Analyze a double-precision operation
// Check if float precision would suffice, and if so, also check if fp16 would
// work
static void analyzeDoubleOperation(int32_t Opcode, double Left, double Right,
                                   double Result, int32_t Id) {
  std::lock_guard<std::mutex> Lock(getStatsMutex());

  OperationStats &Stats = getOperationStats()[Id];
  Stats.TotalCount++;

  // Check for special values in inputs or result
  if (isSpecialValue(Result) || isSpecialValue(Left) || isSpecialValue(Right)) {
    Stats.InputSpecialValues++;
    return;
  }

  // First, try double → float
  double FloatResult = simulateLowerPrecisionOp(Opcode, Left, Right);

  // Check if lowering to float created special values (overflow/underflow)
  if (isSpecialValue(FloatResult)) {
    Stats.DoubleNeedsDouble++;     // Float doesn't work, need to keep double
    Stats.DoubleLoweringSpecial++; // Record that overflow occurred
    return;
  }

  // Compare double vs float results
  double FloatError = computeRelativeError(Result, FloatResult);

  if (FloatError >= DefaultRelativeErrorThreshold) {
    // Float precision is not sufficient, need double
    Stats.DoubleNeedsDouble++;
    return;
  }

  // Float precision is acceptable. Now check if fp16 would also work.
  // Convert operands to float, then simulate fp16 operation
  float LeftF = static_cast<float>(Left);
  float RightF = static_cast<float>(Right);
  float ResultF = static_cast<float>(Result);

  float Fp16Result = simulateFp16Op(Opcode, LeftF, RightF);

  // Check if lowering to fp16 created special values
  if (isSpecialValue(Fp16Result)) {
    // fp16 causes overflow/underflow, but float works (double → float)
    Stats.DoubleToFloat++;         // Float is the lowest we can go
    Stats.DoubleLoweringSpecial++; // Record that fp16 overflow occurred
    return;
  }

  // Compare float vs fp16 results
  double Fp16Error = computeRelativeError(static_cast<double>(ResultF),
                                          static_cast<double>(Fp16Result));

  if (Fp16Error < DefaultRelativeErrorThreshold) {
    // fp16 precision is sufficient (double → fp16)
    Stats.DoubleToFp16++;
  } else {
    // Need float precision but not double (double → float)
    Stats.DoubleToFloat++;
  }
}

// Analyze a float-precision operation (check if half precision would work)
static void analyzeFloatOperation(int32_t Opcode, float Left, float Right,
                                  float Result, int32_t Id) {
  std::lock_guard<std::mutex> Lock(getStatsMutex());

  OperationStats &Stats = getOperationStats()[Id];
  Stats.TotalCount++;

  // Check for special values in inputs or result
  if (isSpecialValue(Result) || isSpecialValue(Left) || isSpecialValue(Right)) {
    Stats.InputSpecialValues++;
    return;
  }

  // Simulate operation with fp16 precision
  float LowerPrecisionResult = simulateFp16Op(Opcode, Left, Right);

  // Check if lowering precision created special values (overflow/underflow to
  // inf)
  if (isSpecialValue(LowerPrecisionResult)) {
    Stats.FloatNeedsFloat++;      // FP16 doesn't work, need to keep float
    Stats.FloatLoweringSpecial++; // Record that overflow occurred
    return;
  }

  // Compare results
  double RelativeError = computeRelativeError(
      static_cast<double>(Result), static_cast<double>(LowerPrecisionResult));

  if (RelativeError < DefaultRelativeErrorThreshold) {
    // fp16 precision is sufficient (float → fp16)
    Stats.FloatToFp16++;
  } else {
    // Need to keep float precision (float → float)
    Stats.FloatNeedsFloat++;
  }
}

extern "C" {

__attribute__((destructor(1000))) void __precision_analysis_finalize() {
  std::printf("\n");
  std::printf("================================================================"
              "==========\n");
  std::printf("            Floating-Point Precision Analysis Results\n");
  std::printf("================================================================"
              "==========\n");
  std::printf(
      "This analysis checks minimum precision needed (error < %.2f%%):\n",
      DefaultRelativeErrorThreshold * 100);
  std::printf("  - Double operations: Try Float, then FP16 if Float works\n");
  std::printf("  - Float operations: Try FP16\n");
  std::printf("================================================================"
              "==========\n\n");

  std::map<int32_t, OperationStats> &OperationStatsMap = getOperationStats();

  if (OperationStatsMap.empty()) {
    std::printf("No operations analyzed.\n");
    std::printf("=============================================================="
                "============\n");
    return;
  }

  uint64_t TotalOps = 0;
  uint64_t TotalDoubleToFp16 = 0;
  uint64_t TotalDoubleToFloat = 0;
  uint64_t TotalDoubleNeedsDouble = 0;
  uint64_t TotalFloatToFp16 = 0;
  uint64_t TotalFloatNeedsFloat = 0;
  uint64_t TotalInputSpecial = 0;
  uint64_t TotalDoubleLoweringSpecial = 0;
  uint64_t TotalFloatLoweringSpecial = 0;

  std::printf("Per-Operation Results:\n");
  std::printf("%-5s %8s %9s %8s %6s %9s %6s %8s %7s %7s\n", "Op ID", "Total",
              "D->FP16", "D->F32", "D->D", "F->FP16", "F->F", "InpNaN",
              "D-OvFl", "F-OvFl");
  std::printf(
      "-------------------------------------------------------------------"
      "-------------\n");

  for (const auto &Entry : OperationStatsMap) {
    int32_t OpId = Entry.first;
    const OperationStats &Stats = Entry.second;

    TotalOps += Stats.TotalCount;
    TotalDoubleToFp16 += Stats.DoubleToFp16;
    TotalDoubleToFloat += Stats.DoubleToFloat;
    TotalDoubleNeedsDouble += Stats.DoubleNeedsDouble;
    TotalFloatToFp16 += Stats.FloatToFp16;
    TotalFloatNeedsFloat += Stats.FloatNeedsFloat;
    TotalInputSpecial += Stats.InputSpecialValues;
    TotalDoubleLoweringSpecial += Stats.DoubleLoweringSpecial;
    TotalFloatLoweringSpecial += Stats.FloatLoweringSpecial;

    std::printf("%-5d %8llu %9llu %8llu %6llu %9llu %6llu %8llu %7llu %7llu\n",
                OpId, Stats.TotalCount, Stats.DoubleToFp16, Stats.DoubleToFloat,
                Stats.DoubleNeedsDouble, Stats.FloatToFp16,
                Stats.FloatNeedsFloat, Stats.InputSpecialValues,
                Stats.DoubleLoweringSpecial, Stats.FloatLoweringSpecial);
  }

  std::printf(
      "-------------------------------------------------------------------"
      "-------------\n");
  std::printf("%-5s %8llu %9llu %8llu %6llu %9llu %6llu %8llu %7llu %7llu\n",
              "TOTAL", TotalOps, TotalDoubleToFp16, TotalDoubleToFloat,
              TotalDoubleNeedsDouble, TotalFloatToFp16, TotalFloatNeedsFloat,
              TotalInputSpecial, TotalDoubleLoweringSpecial,
              TotalFloatLoweringSpecial);

  std::printf("\n");
  std::printf("Column Legend:\n");
  std::printf("  D->FP16:  Double ops that can use FP16 (16-bit)\n");
  std::printf(
      "  D->F32:   Double ops that can use Float (32-bit) but not FP16\n");
  std::printf("  D->D:     Double ops that require Double (64-bit)\n");
  std::printf("  F->FP16:  Float ops that can use FP16 (16-bit)\n");
  std::printf("  F->F:     Float ops that must stay Float (32-bit)\n");
  std::printf("  InpNaN:   Operations with NaN/Inf in inputs or result\n");
  std::printf("  D-OvFl:   Double ops where lowering caused overflow\n");
  std::printf("  F-OvFl:   Float ops where lowering to FP16 caused overflow\n");

  uint64_t TotalDoubleOps =
      TotalDoubleToFp16 + TotalDoubleToFloat + TotalDoubleNeedsDouble;
  uint64_t TotalFloatOps = TotalFloatToFp16 + TotalFloatNeedsFloat;
  uint64_t AnalyzedTotal = TotalDoubleOps + TotalFloatOps;

  std::printf("\n");
  std::printf("================================================================"
              "==========\n");
  std::printf("Summary by Original Precision:\n");
  std::printf("================================================================"
              "==========\n");

  if (TotalDoubleOps > 0) {
    std::printf("\nDOUBLE Operations (started as 64-bit double):\n");
    std::printf("  Total:                              %llu\n", TotalDoubleOps);
    std::printf("  Can reduce to FP16 (16-bit):        %llu (%.1f%%)\n",
                TotalDoubleToFp16, 100.0 * TotalDoubleToFp16 / TotalDoubleOps);
    std::printf("  Can reduce to Float (32-bit):       %llu (%.1f%%)\n",
                TotalDoubleToFloat,
                100.0 * TotalDoubleToFloat / TotalDoubleOps);
    std::printf("  Must keep Double (64-bit):          %llu (%.1f%%)\n",
                TotalDoubleNeedsDouble,
                100.0 * TotalDoubleNeedsDouble / TotalDoubleOps);

    uint64_t DoubleConvertible = TotalDoubleToFp16 + TotalDoubleToFloat;
    std::printf("  → Total convertible to lower:       %llu (%.1f%%)\n",
                DoubleConvertible, 100.0 * DoubleConvertible / TotalDoubleOps);
  }

  if (TotalFloatOps > 0) {
    std::printf("\nFLOAT Operations (started as 32-bit float):\n");
    std::printf("  Total:                              %llu\n", TotalFloatOps);
    std::printf("  Can reduce to FP16 (16-bit):        %llu (%.1f%%)\n",
                TotalFloatToFp16, 100.0 * TotalFloatToFp16 / TotalFloatOps);
    std::printf("  Must keep Float (32-bit):           %llu (%.1f%%)\n",
                TotalFloatNeedsFloat,
                100.0 * TotalFloatNeedsFloat / TotalFloatOps);
  }

  std::printf("\nOVERALL Statistics:\n");
  std::printf("  Total analyzed operations:          %llu\n", AnalyzedTotal);
  std::printf("  Operations with input NaN/Inf:      %llu\n",
              TotalInputSpecial);
  std::printf("  Double ops causing overflow:        %llu\n",
              TotalDoubleLoweringSpecial);
  std::printf("  Float ops causing overflow:         %llu\n",
              TotalFloatLoweringSpecial);

  if (AnalyzedTotal > 0) {
    uint64_t TotalToFp16 = TotalDoubleToFp16 + TotalFloatToFp16;
    std::printf("\n  ALL operations reducible to FP16:   %llu (%.1f%%)\n",
                TotalToFp16, 100.0 * TotalToFp16 / AnalyzedTotal);
  }

  // Provide recommendations based on results
  std::printf("\n=============================================================="
              "============\n");
  std::printf("Recommendations:\n");
  std::printf("================================================================"
              "==========\n");

  if (TotalDoubleOps > 0) {
    // Include overflow operations in total for realistic assessment
    uint64_t TotalDoubleWithOverflow =
        TotalDoubleOps + TotalDoubleLoweringSpecial;
    double DoubleToLower = 100.0 * (TotalDoubleToFp16 + TotalDoubleToFloat) /
                           TotalDoubleWithOverflow;
    double OverflowPct =
        100.0 * TotalDoubleLoweringSpecial / TotalDoubleWithOverflow;

    std::printf("\nFor DOUBLE operations:\n");
    std::printf("  Analyzed: %llu (%.1f%% overflow, not convertible)\n",
                TotalDoubleWithOverflow, OverflowPct);

    if (DoubleToLower > 80.0) {
      std::printf(
          "  ✓ %.1f%% can use lower precision - strong conversion candidate\n",
          DoubleToLower);
      if (TotalDoubleToFp16 > TotalDoubleToFloat) {
        std::printf("  ✓ Many can go directly to FP16 - consider aggressive "
                    "downcasting\n");
      } else {
        std::printf(
            "  ✓ Most need Float - consider using f32 instead of f64\n");
      }
      if (TotalDoubleLoweringSpecial > 0 && OverflowPct > 5.0) {
        std::printf(
            "  ⚠ %.1f%% overflow - may need value scaling/normalization\n",
            OverflowPct);
      }
    } else if (DoubleToLower > 50.0) {
      std::printf(
          "  ~ %.1f%% can use lower precision - mixed precision recommended\n",
          DoubleToLower);
      if (TotalDoubleLoweringSpecial > 0) {
        std::printf("  ⚠ %.1f%% overflow - limits conversion opportunities\n",
                    OverflowPct);
      }
    } else {
      std::printf("  ✗ Only %.1f%% can use lower precision - keep double\n",
                  DoubleToLower);
      if (TotalDoubleLoweringSpecial > TotalDoubleNeedsDouble) {
        std::printf("  ! Most failures due to overflow (%.1f%%) rather than "
                    "accuracy (%llu ops)\n",
                    OverflowPct, TotalDoubleNeedsDouble);
        std::printf("  → Problem is value range, not precision\n");
      }
    }
  }

  if (TotalFloatOps > 0) {
    // Include overflow operations in total for realistic assessment
    uint64_t TotalFloatWithOverflow = TotalFloatOps + TotalFloatLoweringSpecial;
    double FloatToFp16Pct = 100.0 * TotalFloatToFp16 / TotalFloatWithOverflow;
    double FloatOverflowPct =
        100.0 * TotalFloatLoweringSpecial / TotalFloatWithOverflow;

    std::printf("\nFor FLOAT operations:\n");
    std::printf("  Analyzed: %llu (%.1f%% overflow to FP16)\n",
                TotalFloatWithOverflow, FloatOverflowPct);

    if (FloatToFp16Pct > 80.0) {
      std::printf(
          "  ✓ %.1f%% can use FP16 - strong FP16 conversion candidate\n",
          FloatToFp16Pct);
      if (TotalFloatLoweringSpecial > 0 && FloatOverflowPct > 5.0) {
        std::printf("  ⚠ %.1f%% overflow (values exceed FP16 range ±65504)\n",
                    FloatOverflowPct);
      }
    } else if (FloatToFp16Pct > 50.0) {
      std::printf("  ~ %.1f%% can use FP16 - selective FP16 use recommended\n",
                  FloatToFp16Pct);
      if (TotalFloatLoweringSpecial > 0) {
        std::printf("  ⚠ %.1f%% overflow - limits FP16 opportunities\n",
                    FloatOverflowPct);
      }
    } else {
      std::printf("  ✗ Only %.1f%% can use FP16 - keep float\n",
                  FloatToFp16Pct);
      if (TotalFloatLoweringSpecial > TotalFloatNeedsFloat) {
        std::printf("  ! Most failures due to FP16 overflow (%.1f%%) rather "
                    "than accuracy (%llu ops)\n",
                    FloatOverflowPct, TotalFloatNeedsFloat);
        std::printf("  → Problem: Values exceed FP16 range (±65504)\n");
        std::printf("  → Solution: Scale values or use Float\n");
      }
    }
  }

  std::printf("================================================================"
              "==========\n");
}

void __precision_analysis_post_numeric(int32_t type_id, int32_t sub_type_id,
                                       int32_t size, int32_t opcode,
                                       int64_t left, int64_t right,
                                       int64_t result, int64_t flags,
                                       int32_t id) {
  // Handle vector types by looking at sub_type_id
  bool IsVector = false;
  int32_t ElementTypeId = type_id;

  switch (type_id) {
  case FixedVectorTyID:
  case ScalableVectorTyID:
    IsVector = true;
    ElementTypeId = sub_type_id;
    break;
  default:
    break;
  }

  // For vector operations, we'd need to extract each element
  // For now, skip vector operations (they're more complex)
  if (IsVector) {
    return;
  }

  // Analyze based on type
  if (ElementTypeId == DoubleTyID) {
    // Double precision operation - check if float would suffice
    double LeftVal = *reinterpret_cast<double *>(&left);
    double RightVal = *reinterpret_cast<double *>(&right);
    double ResultVal = *reinterpret_cast<double *>(&result);

    analyzeDoubleOperation(opcode, LeftVal, RightVal, ResultVal, id);
  } else if (ElementTypeId == FloatTyID) {
    // Float precision operation - could check if half would suffice
    float LeftVal = *reinterpret_cast<float *>(&left);
    float RightVal = *reinterpret_cast<float *>(&right);
    float ResultVal = *reinterpret_cast<float *>(&result);

    analyzeFloatOperation(opcode, LeftVal, RightVal, ResultVal, id);
  }
  // Skip other types (half, bfloat, extended precision)
}

void __precision_analysis_post_numeric_ind(int32_t type_id, int32_t sub_type_id,
                                           int32_t size, int32_t opcode,
                                           int64_t *left_ptr,
                                           int64_t *right_ptr,
                                           int64_t *result_ptr, int64_t flags,
                                           int32_t id) {}

} // extern "C"
