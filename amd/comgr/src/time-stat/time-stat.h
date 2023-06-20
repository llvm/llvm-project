#ifndef AMD_COMGR_TIME_STAT_H
#define AMD_COMGR_TIME_STAT_H

#include "perf-timer.h"
#include "llvm/ADT/StringMap.h"

#include "amd_comgr.h"
#include <iostream>

namespace TimeStatistics {

struct ProfileData {
  double TimeTaken;
  int Counter;
};

class PerfStats {
  std::unique_ptr<llvm::raw_fd_ostream,
                  std::function<void(llvm::raw_fd_ostream *)>>
      pLog;
  PerfTimer PT;

  llvm::StringMap<ProfileData> ProfileDataMap;

public:
  PerfStats() {}
  bool Init(std::string LogFile) {
    std::error_code EC;
    std::unique_ptr<llvm::raw_fd_ostream,
                    std::function<void(llvm::raw_fd_ostream *)>>
        LogF(new (std::nothrow)
                 llvm::raw_fd_ostream(LogFile, EC, llvm::sys::fs::OF_Text),
             [](llvm::raw_fd_ostream *fp) {
               *fp << "Closing log...\n";
               fp->close();
             });
    if (EC) {
      std::cerr << "Failed to open log file " << LogFile << "for perf stats "
                << EC.message() << "\n ";
      return false;
    } else {
      pLog = std::move(LogF);
    }

    // Initialize Timer
    if (!PT.Init())
      return false;

    return true;
  }

  double getCurrentTime() { return PT.getCurrentTime(); }

  void AddToStats(llvm::StringRef Name, double TimeTaken) {
    ProfileDataMap[Name].TimeTaken += TimeTaken;
    ProfileDataMap[Name].Counter++;
  }

  void dumpPerfStats() {
    for (const auto &Item : ProfileDataMap) {
      *pLog << "Profile Point "
            << llvm::format("%-50s", Item.getKey().str().c_str())
            << " was invoked " << llvm::format("%6d", Item.getValue().Counter)
            << " times and took "
            << llvm::format("%10.4f", Item.getValue().TimeTaken)
            << " milliseconds overall\n";
    }
  }
};

} // namespace TimeStatistics

#endif // AMD_COMGR_TIME_STAT_H
