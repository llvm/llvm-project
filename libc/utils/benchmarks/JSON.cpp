//===-- JSON serialization routines ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "JSON.h"
#include "LibcBenchmark.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MathExtras.h"

#include <chrono>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace llvm {
namespace libc_benchmarks {

template <typename T>
static Error intFromJsonTemplate(const json::Value &V, T &Out) {
  if (const auto &MaybeInt64 = V.getAsInteger()) {
    int64_t Value = *MaybeInt64;
    if (Value < std::numeric_limits<T>::min() ||
        Value > std::numeric_limits<T>::max())
      return createStringError(errc::io_error, "Out of bound Integer");
    Out = Value;
    return Error::success();
  }
  return createStringError(errc::io_error, "Can't parse Integer");
}

static Error fromJson(const json::Value &V, double &Out) {
  if (auto S = V.getAsNumber()) {
    Out = *S;
    return Error::success();
  }
  return createStringError(errc::io_error, "Can't parse Double");
}

static Error fromJson(const json::Value &V, std::string &Out) {
  if (auto S = V.getAsString()) {
    Out = std::string(*S);
    return Error::success();
  }
  return createStringError(errc::io_error, "Can't parse String");
}

static Error fromJson(const json::Value &V, uint32_t &Out) {
  return intFromJsonTemplate(V, Out);
}

static Error fromJson(const json::Value &V, uint8_t &Out) {
  return intFromJsonTemplate(V, Out);
}

static Error fromJson(const json::Value &V, int &Out) {
  return intFromJsonTemplate(V, Out);
}

static Error fromJson(const json::Value &V, libc_benchmarks::Duration &D) {
  if (V.kind() != json::Value::Kind::Number)
    return createStringError(errc::io_error, "Can't parse Duration");
  D = libc_benchmarks::Duration(*V.getAsNumber());
  return Error::success();
}

static Error fromJson(const json::Value &V, MaybeAlign &Out) {
  const auto MaybeInt = V.getAsInteger();
  if (!MaybeInt)
    return createStringError(errc::io_error,
                             "Can't parse Align, not an Integer");
  const int64_t Value = *MaybeInt;
  if (!Value) {
    Out = None;
    return Error::success();
  }
  if (isPowerOf2_64(Value)) {
    Out = Align(Value);
    return Error::success();
  }
  return createStringError(errc::io_error,
                           "Can't parse Align, not a power of two");
}

static Error fromJson(const json::Value &V,
                      libc_benchmarks::BenchmarkLog &Out) {
  if (V.kind() != json::Value::Kind::String)
    return createStringError(errc::io_error,
                             "Can't parse BenchmarkLog, not a String");
  const auto String = *V.getAsString();
  auto Parsed =
      llvm::StringSwitch<Optional<libc_benchmarks::BenchmarkLog>>(String)
          .Case("None", libc_benchmarks::BenchmarkLog::None)
          .Case("Last", libc_benchmarks::BenchmarkLog::Last)
          .Case("Full", libc_benchmarks::BenchmarkLog::Full)
          .Default(None);
  if (!Parsed)
    return createStringError(errc::io_error,
                             Twine("Can't parse BenchmarkLog, invalid value '")
                                 .concat(String)
                                 .concat("'"));
  Out = *Parsed;
  return Error::success();
}

template <typename C>
Error vectorFromJsonTemplate(const json::Value &V, C &Out) {
  auto *A = V.getAsArray();
  if (!A)
    return createStringError(errc::io_error, "Can't parse Array");
  Out.clear();
  Out.resize(A->size());
  for (auto InOutPair : llvm::zip(*A, Out))
    if (auto E = fromJson(std::get<0>(InOutPair), std::get<1>(InOutPair)))
      return std::move(E);
  return Error::success();
}

template <typename T>
static Error fromJson(const json::Value &V, std::vector<T> &Out) {
  return vectorFromJsonTemplate(V, Out);
}

template <typename T>
static Error fromJson(const json::Value &V, SmallVectorImpl<T> &Out) {
  return vectorFromJsonTemplate(V, Out);
}

// Same as llvm::json::ObjectMapper but adds a finer error reporting mechanism.
class JsonObjectMapper {
  const json::Object *O;
  Error E;
  SmallDenseSet<StringRef> SeenFields;

public:
  explicit JsonObjectMapper(const json::Value &V)
      : O(V.getAsObject()),
        E(O ? Error::success()
            : createStringError(errc::io_error, "Expected JSON Object")) {}

  Error takeError() {
    if (E)
      return std::move(E);
    for (const auto &Itr : *O) {
      const StringRef Key = Itr.getFirst();
      if (!SeenFields.count(Key))
        E = createStringError(errc::io_error,
                              Twine("Unknown field: ").concat(Key));
    }
    return std::move(E);
  }

  template <typename T> void map(StringRef Key, T &Out) {
    if (E)
      return;
    if (const json::Value *Value = O->get(Key)) {
      SeenFields.insert(Key);
      E = fromJson(*Value, Out);
    }
  }
};

static Error fromJson(const json::Value &V,
                      libc_benchmarks::BenchmarkOptions &Out) {
  JsonObjectMapper O(V);
  O.map("MinDuration", Out.MinDuration);
  O.map("MaxDuration", Out.MaxDuration);
  O.map("InitialIterations", Out.InitialIterations);
  O.map("MaxIterations", Out.MaxIterations);
  O.map("MinSamples", Out.MinSamples);
  O.map("MaxSamples", Out.MaxSamples);
  O.map("Epsilon", Out.Epsilon);
  O.map("ScalingFactor", Out.ScalingFactor);
  O.map("Log", Out.Log);
  return O.takeError();
}

static Error fromJson(const json::Value &V, libc_benchmarks::SizeRange &Out) {
  JsonObjectMapper O(V);
  O.map("From", Out.From);
  O.map("To", Out.To);
  O.map("Step", Out.Step);
  return O.takeError();
}

static Error fromJson(const json::Value &V,
                      libc_benchmarks::StudyConfiguration &Out) {
  JsonObjectMapper O(V);
  O.map("Runs", Out.Runs);
  O.map("BufferSize", Out.BufferSize);
  O.map("Size", Out.Size);
  O.map("AddressAlignment", Out.AddressAlignment);
  O.map("MemsetValue", Out.MemsetValue);
  O.map("MemcmpMismatchAt", Out.MemcmpMismatchAt);
  return O.takeError();
}

static Error fromJson(const json::Value &V, libc_benchmarks::CacheInfo &Out) {
  JsonObjectMapper O(V);
  O.map("Type", Out.Type);
  O.map("Level", Out.Level);
  O.map("Size", Out.Size);
  O.map("NumSharing", Out.NumSharing);
  return O.takeError();
}

static Error fromJson(const json::Value &V, libc_benchmarks::HostState &Out) {
  JsonObjectMapper O(V);
  O.map("CpuName", Out.CpuName);
  O.map("CpuFrequency", Out.CpuFrequency);
  O.map("Caches", Out.Caches);
  return O.takeError();
}

static Error fromJson(const json::Value &V,
                      libc_benchmarks::FunctionMeasurements &Out) {
  JsonObjectMapper O(V);
  O.map("Name", Out.Name);
  std::vector<uint32_t> Sizes;
  O.map("Sizes", Sizes);
  std::vector<libc_benchmarks::Duration> Runtimes;
  O.map("Runtimes", Runtimes);
  if (Sizes.size() != Runtimes.size())
    return createStringError(errc::io_error,
                             "Measurement Size and Runtime mistmatch");
  Out.Measurements.resize(Sizes.size());
  for (size_t I = 0; I < Sizes.size(); ++I) {
    Out.Measurements[I].Size = Sizes[I];
    Out.Measurements[I].Runtime = Runtimes[I];
  }
  return O.takeError();
}

static Error fromJson(const json::Value &V, libc_benchmarks::Study &Out) {
  JsonObjectMapper O(V);
  O.map("Host", Out.Host);
  O.map("Options", Out.Options);
  O.map("Configuration", Out.Configuration);
  O.map("Functions", Out.Functions);
  return O.takeError();
}

static double Seconds(const Duration &D) {
  return std::chrono::duration<double>(D).count();
}

Expected<Study> ParseJsonStudy(StringRef Content) {
  Expected<json::Value> EV = json::parse(Content);
  if (!EV)
    return EV.takeError();
  Study S;
  if (Error E = fromJson(*EV, S))
    return std::move(E);
  return S;
}

static StringRef Serialize(const BenchmarkLog &L) {
  switch (L) {
  case BenchmarkLog::None:
    return "None";
  case BenchmarkLog::Last:
    return "Last";
  case BenchmarkLog::Full:
    return "Full";
  }
  llvm_unreachable("Unhandled BenchmarkLog value");
}

static void Serialize(const BenchmarkOptions &BO, json::OStream &JOS) {
  JOS.object([&]() {
    JOS.attribute("MinDuration", Seconds(BO.MinDuration));
    JOS.attribute("MaxDuration", Seconds(BO.MaxDuration));
    JOS.attribute("InitialIterations", BO.InitialIterations);
    JOS.attribute("MaxIterations", BO.MaxIterations);
    JOS.attribute("MinSamples", BO.MinSamples);
    JOS.attribute("MaxSamples", BO.MaxSamples);
    JOS.attribute("Epsilon", BO.Epsilon);
    JOS.attribute("ScalingFactor", BO.ScalingFactor);
    JOS.attribute("Log", Serialize(BO.Log));
  });
}

static void Serialize(const CacheInfo &CI, json::OStream &JOS) {
  JOS.object([&]() {
    JOS.attribute("Type", CI.Type);
    JOS.attribute("Level", CI.Level);
    JOS.attribute("Size", CI.Size);
    JOS.attribute("NumSharing", CI.NumSharing);
  });
}

static void Serialize(const HostState &HS, json::OStream &JOS) {
  JOS.object([&]() {
    JOS.attribute("CpuName", HS.CpuName);
    JOS.attribute("CpuFrequency", HS.CpuFrequency);
    JOS.attributeArray("Caches", [&]() {
      for (const auto &CI : HS.Caches)
        Serialize(CI, JOS);
    });
  });
}

static void Serialize(const StudyConfiguration &SC, json::OStream &JOS) {
  JOS.object([&]() {
    JOS.attribute("Runs", SC.Runs);
    JOS.attribute("BufferSize", SC.BufferSize);
    JOS.attributeObject("Size", [&]() {
      JOS.attribute("From", SC.Size.From);
      JOS.attribute("To", SC.Size.To);
      JOS.attribute("Step", SC.Size.Step);
    });
    if (SC.AddressAlignment)
      JOS.attribute("AddressAlignment",
                    static_cast<int64_t>(SC.AddressAlignment->value()));
    JOS.attribute("MemsetValue", SC.MemsetValue);
    JOS.attribute("MemcmpMismatchAt", SC.MemcmpMismatchAt);
  });
}

static void Serialize(const FunctionMeasurements &FM, json::OStream &JOS) {
  JOS.object([&]() {
    JOS.attribute("Name", FM.Name);
    JOS.attributeArray("Sizes", [&]() {
      for (const auto &M : FM.Measurements)
        JOS.value(M.Size);
    });
    JOS.attributeArray("Runtimes", [&]() {
      for (const auto &M : FM.Measurements)
        JOS.value(Seconds(M.Runtime));
    });
  });
}

void SerializeToJson(const Study &S, json::OStream &JOS) {
  JOS.object([&]() {
    JOS.attributeBegin("Host");
    Serialize(S.Host, JOS);
    JOS.attributeEnd();

    JOS.attributeBegin("Options");
    Serialize(S.Options, JOS);
    JOS.attributeEnd();

    JOS.attributeBegin("Configuration");
    Serialize(S.Configuration, JOS);
    JOS.attributeEnd();

    if (!S.Functions.empty()) {
      JOS.attributeArray("Functions", [&]() {
        for (const auto &FM : S.Functions)
          Serialize(FM, JOS);
      });
    }
  });
}

} // namespace libc_benchmarks
} // namespace llvm
