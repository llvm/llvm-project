//===- bolt/Core/DynoStats.h - Dynamic execution stats ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Keep track of statistics about the trace of execution captured in BOLT
// profile.
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_CORE_DYNO_STATS_H
#define BOLT_CORE_DYNO_STATS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/Support/raw_ostream.h"
#include <map>
#include <unordered_map>

namespace llvm {

namespace bolt {
class BinaryFunction;

/// Class encapsulating runtime statistics about an execution unit.
class DynoStats {

#define REAL_DYNO_STATS                                                        \
  D(FORWARD_COND_BRANCHES, "executed forward branches", Fn)                    \
  D(FORWARD_COND_BRANCHES_TAKEN, "taken forward branches", Fn)                 \
  D(BACKWARD_COND_BRANCHES, "executed backward branches", Fn)                  \
  D(BACKWARD_COND_BRANCHES_TAKEN, "taken backward branches", Fn)               \
  D(UNCOND_BRANCHES, "executed unconditional branches", Fn)                    \
  D(FUNCTION_CALLS, "all function calls", Fn)                                  \
  D(INDIRECT_CALLS, "indirect calls", Fn)                                      \
  D(PLT_CALLS, "PLT calls", Fn)                                                \
  D(INSTRUCTIONS, "executed instructions", Fn)                                 \
  D(LOADS, "executed load instructions", Fn)                                   \
  D(STORES, "executed store instructions", Fn)                                 \
  D(JUMP_TABLE_BRANCHES, "taken jump table branches", Fn)                      \
  D(UNKNOWN_INDIRECT_BRANCHES, "taken unknown indirect branches", Fn)          \
  D(ALL_BRANCHES, "total branches", Fadd(ALL_CONDITIONAL, UNCOND_BRANCHES))    \
  D(ALL_TAKEN, "taken branches", Fadd(TAKEN_CONDITIONAL, UNCOND_BRANCHES))     \
  D(NONTAKEN_CONDITIONAL, "non-taken conditional branches",                    \
    Fsub(ALL_CONDITIONAL, TAKEN_CONDITIONAL))                                  \
  D(TAKEN_CONDITIONAL, "taken conditional branches",                           \
    Fadd(FORWARD_COND_BRANCHES_TAKEN, BACKWARD_COND_BRANCHES_TAKEN))           \
  D(ALL_CONDITIONAL, "all conditional branches",                               \
    Fadd(FORWARD_COND_BRANCHES, BACKWARD_COND_BRANCHES))                       \
  D(VENEER_CALLS_AARCH64, "linker-inserted veneer calls", Fn)

#define DYNO_STATS                                                             \
  D(FIRST_DYNO_STAT, "<reserved>", 0)                                          \
  REAL_DYNO_STATS                                                              \
  D(LAST_DYNO_STAT, "<reserved>", 0)

public:
#define D(name, ...) name,
  enum Category : uint8_t { DYNO_STATS };
#undef D

private:
  uint64_t Stats[LAST_DYNO_STAT + 1];
  bool PrintAArch64Stats;

#define D(name, desc, ...) desc,
  static constexpr const char *Desc[] = { DYNO_STATS };
#undef D

public:
  DynoStats(bool PrintAArch64Stats) {
    this->PrintAArch64Stats = PrintAArch64Stats;
    for (auto Stat = FIRST_DYNO_STAT + 0; Stat < LAST_DYNO_STAT; ++Stat)
      Stats[Stat] = 0;
  }

  uint64_t &operator[](size_t I) {
    assert(I > FIRST_DYNO_STAT && I < LAST_DYNO_STAT && "index out of bounds");
    return Stats[I];
  }

  uint64_t operator[](size_t I) const {
    switch (I) {
#define D(name, desc, func) \
    case name: \
      return func;
#define Fn Stats[I]
#define Fadd(a, b) operator[](a) + operator[](b)
#define Fsub(a, b) operator[](a) - operator[](b)
#define F(a) operator[](a)
#define Radd(a, b) (a + b)
#define Rsub(a, b) (a - b)
    DYNO_STATS
#undef Rsub
#undef Radd
#undef F
#undef Fsub
#undef Fadd
#undef Fn
#undef D
    default:
      llvm_unreachable("index out of bounds");
    }
    return 0;
  }

  void print(raw_ostream &OS, const DynoStats *Other = nullptr,
             MCInstPrinter *Printer = nullptr) const;

  void operator+=(const DynoStats &Other);
  bool operator<(const DynoStats &Other) const;
  bool operator==(const DynoStats &Other) const;
  bool operator!=(const DynoStats &Other) const { return !operator==(Other); }
  bool lessThan(const DynoStats &Other, ArrayRef<Category> Keys) const;

  static const char *Description(const Category C) { return Desc[C]; }

  /// Maps instruction opcodes to:
  /// 1. Accumulated executed instruction counts.
  /// 2. a multimap that records highest execution counts, function names,
  /// and BB offsets where instructions of these opcodes occur.
  using MaxOpcodeHistogramTy =
      std::multimap<uint64_t, std::pair<StringRef, uint32_t>>;
  using OpcodeHistogramTy =
      std::unordered_map<unsigned, std::pair<uint64_t, MaxOpcodeHistogramTy>>;
  using OpcodeStatTy = OpcodeHistogramTy::value_type;

  OpcodeHistogramTy OpcodeHistogram;
};

inline raw_ostream &operator<<(raw_ostream &OS, const DynoStats &Stats) {
  Stats.print(OS, nullptr);
  return OS;
}

DynoStats operator+(const DynoStats &A, const DynoStats &B);

/// Return dynostats for the function.
///
/// The function relies on branch instructions being in-sync with CFG for
/// branch instructions stats. Thus it is better to call it after
/// fixBranches().
DynoStats getDynoStats(BinaryFunction &BF);

/// Return program-wide dynostats.
template <typename FuncsType>
inline DynoStats getDynoStats(FuncsType &Funcs, bool IsAArch64) {
  DynoStats dynoStats(IsAArch64);
  for (auto &BFI : Funcs) {
    auto &BF = BFI.second;
    if (BF.isSimple())
      dynoStats += getDynoStats(BF);
  }
  return dynoStats;
}

/// Call a function with optional before and after dynostats printing.
template <typename FnType, typename FuncsType>
inline void callWithDynoStats(raw_ostream &OS, FnType &&Func, FuncsType &Funcs,
                              StringRef Phase, const bool Flag,
                              bool IsAArch64) {
  DynoStats DynoStatsBefore(IsAArch64);
  if (Flag)
    DynoStatsBefore = getDynoStats(Funcs, IsAArch64);

  Func();

  if (Flag) {
    const DynoStats DynoStatsAfter = getDynoStats(Funcs, IsAArch64);
    const bool Changed = (DynoStatsAfter != DynoStatsBefore);
    OS << "BOLT-INFO: program-wide dynostats after running " << Phase
       << (Changed ? "" : " (no change)") << ":\n\n"
       << DynoStatsBefore << '\n';
    if (Changed)
      DynoStatsAfter.print(OS, &DynoStatsBefore);
    OS << '\n';
  }
}

} // namespace bolt
} // namespace llvm

#endif
