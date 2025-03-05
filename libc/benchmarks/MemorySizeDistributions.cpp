#include "MemorySizeDistributions.h"

#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {
namespace libc_benchmarks {

static constexpr double MemmoveGoogleA[] = {
#include "distributions/MemmoveGoogleA.csv"
};
static constexpr double MemmoveGoogleB[] = {
#include "distributions/MemmoveGoogleB.csv"
};
static constexpr double MemmoveGoogleD[] = {
#include "distributions/MemmoveGoogleD.csv"
};
static constexpr double MemmoveGoogleQ[] = {
#include "distributions/MemmoveGoogleQ.csv"
};
static constexpr double MemmoveGoogleL[] = {
#include "distributions/MemmoveGoogleL.csv"
};
static constexpr double MemmoveGoogleM[] = {
#include "distributions/MemmoveGoogleM.csv"
};
static constexpr double MemmoveGoogleS[] = {
#include "distributions/MemmoveGoogleS.csv"
};
static constexpr double MemmoveGoogleW[] = {
#include "distributions/MemmoveGoogleW.csv"
};
static constexpr double MemmoveGoogleU[] = {
#include "distributions/MemmoveGoogleU.csv"
};
static constexpr double MemcmpGoogleA[] = {
#include "distributions/MemcmpGoogleA.csv"
};
static constexpr double MemcmpGoogleB[] = {
#include "distributions/MemcmpGoogleB.csv"
};
static constexpr double MemcmpGoogleD[] = {
#include "distributions/MemcmpGoogleD.csv"
};
static constexpr double MemcmpGoogleQ[] = {
#include "distributions/MemcmpGoogleQ.csv"
};
static constexpr double MemcmpGoogleL[] = {
#include "distributions/MemcmpGoogleL.csv"
};
static constexpr double MemcmpGoogleM[] = {
#include "distributions/MemcmpGoogleM.csv"
};
static constexpr double MemcmpGoogleS[] = {
#include "distributions/MemcmpGoogleS.csv"
};
static constexpr double MemcmpGoogleW[] = {
#include "distributions/MemcmpGoogleW.csv"
};
static constexpr double MemcmpGoogleU[] = {
#include "distributions/MemcmpGoogleU.csv"
};
static constexpr double MemcpyGoogleA[] = {
#include "distributions/MemcpyGoogleA.csv"
};
static constexpr double MemcpyGoogleB[] = {
#include "distributions/MemcpyGoogleB.csv"
};
static constexpr double MemcpyGoogleD[] = {
#include "distributions/MemcpyGoogleD.csv"
};
static constexpr double MemcpyGoogleQ[] = {
#include "distributions/MemcpyGoogleQ.csv"
};
static constexpr double MemcpyGoogleL[] = {
#include "distributions/MemcpyGoogleL.csv"
};
static constexpr double MemcpyGoogleM[] = {
#include "distributions/MemcpyGoogleM.csv"
};
static constexpr double MemcpyGoogleS[] = {
#include "distributions/MemcpyGoogleS.csv"
};
static constexpr double MemcpyGoogleW[] = {
#include "distributions/MemcpyGoogleW.csv"
};
static constexpr double MemcpyGoogleU[] = {
#include "distributions/MemcpyGoogleU.csv"
};
static constexpr double MemsetGoogleA[] = {
#include "distributions/MemsetGoogleA.csv"
};
static constexpr double MemsetGoogleB[] = {
#include "distributions/MemsetGoogleB.csv"
};
static constexpr double MemsetGoogleD[] = {
#include "distributions/MemsetGoogleD.csv"
};
static constexpr double MemsetGoogleQ[] = {
#include "distributions/MemsetGoogleQ.csv"
};
static constexpr double MemsetGoogleL[] = {
#include "distributions/MemsetGoogleL.csv"
};
static constexpr double MemsetGoogleM[] = {
#include "distributions/MemsetGoogleM.csv"
};
static constexpr double MemsetGoogleS[] = {
#include "distributions/MemsetGoogleS.csv"
};
static constexpr double MemsetGoogleW[] = {
#include "distributions/MemsetGoogleW.csv"
};
static constexpr double MemsetGoogleU[] = {
#include "distributions/MemsetGoogleU.csv"
};
static constexpr double Uniform384To4096[] = {
#include "distributions/Uniform384To4096.csv"
};

ArrayRef<MemorySizeDistribution> getMemmoveSizeDistributions() {
  static constexpr MemorySizeDistribution kDistributions[] = {
      {"memmove Google A", MemmoveGoogleA},
      {"memmove Google B", MemmoveGoogleB},
      {"memmove Google D", MemmoveGoogleD},
      {"memmove Google L", MemmoveGoogleL},
      {"memmove Google M", MemmoveGoogleM},
      {"memmove Google Q", MemmoveGoogleQ},
      {"memmove Google S", MemmoveGoogleS},
      {"memmove Google U", MemmoveGoogleU},
      {"memmove Google W", MemmoveGoogleW},
      {"uniform 384 to 4096", Uniform384To4096},
  };
  return kDistributions;
}

ArrayRef<MemorySizeDistribution> getMemcpySizeDistributions() {
  static constexpr MemorySizeDistribution kDistributions[] = {
      {"memcpy Google A", MemcpyGoogleA},
      {"memcpy Google B", MemcpyGoogleB},
      {"memcpy Google D", MemcpyGoogleD},
      {"memcpy Google L", MemcpyGoogleL},
      {"memcpy Google M", MemcpyGoogleM},
      {"memcpy Google Q", MemcpyGoogleQ},
      {"memcpy Google S", MemcpyGoogleS},
      {"memcpy Google U", MemcpyGoogleU},
      {"memcpy Google W", MemcpyGoogleW},
      {"uniform 384 to 4096", Uniform384To4096},
  };
  return kDistributions;
}

ArrayRef<MemorySizeDistribution> getMemsetSizeDistributions() {
  static constexpr MemorySizeDistribution kDistributions[] = {
      {"memset Google A", MemsetGoogleA},
      {"memset Google B", MemsetGoogleB},
      {"memset Google D", MemsetGoogleD},
      {"memset Google L", MemsetGoogleL},
      {"memset Google M", MemsetGoogleM},
      {"memset Google Q", MemsetGoogleQ},
      {"memset Google S", MemsetGoogleS},
      {"memset Google U", MemsetGoogleU},
      {"memset Google W", MemsetGoogleW},
      {"uniform 384 to 4096", Uniform384To4096},
  };
  return kDistributions;
}

ArrayRef<MemorySizeDistribution> getMemcmpSizeDistributions() {
  static constexpr MemorySizeDistribution kDistributions[] = {
      {"memcmp Google A", MemcmpGoogleA},
      {"memcmp Google B", MemcmpGoogleB},
      {"memcmp Google D", MemcmpGoogleD},
      {"memcmp Google L", MemcmpGoogleL},
      {"memcmp Google M", MemcmpGoogleM},
      {"memcmp Google Q", MemcmpGoogleQ},
      {"memcmp Google S", MemcmpGoogleS},
      {"memcmp Google U", MemcmpGoogleU},
      {"memcmp Google W", MemcmpGoogleW},
      {"uniform 384 to 4096", Uniform384To4096},
  };
  return kDistributions;
}

MemorySizeDistribution
getDistributionOrDie(ArrayRef<MemorySizeDistribution> Distributions,
                     StringRef Name) {
  for (const auto &MSD : Distributions)
    if (MSD.Name == Name)
      return MSD;

  std::string Message;
  raw_string_ostream Stream(Message);
  Stream << "Unknown MemorySizeDistribution '" << Name
         << "', available distributions:\n";
  for (const auto &MSD : Distributions)
    Stream << "'" << MSD.Name << "'\n";
  report_fatal_error(Message);
}

} // namespace libc_benchmarks
} // namespace llvm
