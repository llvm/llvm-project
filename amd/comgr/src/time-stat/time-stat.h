#ifndef AMD_COMGR_TIME_STAT_H
#define AMD_COMGR_TIME_STAT_H

#include "perf-timer.h"

#include "amd_comgr.h"

StringRef getActionKindName(amd_comgr_action_kind_t ActionKind);

namespace TimeStatistics {

typedef struct ActionStat {
  uint16_t ExecCounter = 0;
  double OverallTime = 0.0;
} ActionStat_t;

class PerfStats {
  ActionStat_t Stats[AMD_COMGR_ACTION_LAST + 1];
  std::unique_ptr<raw_fd_ostream, std::function<void(raw_fd_ostream *)>> pLog;
  PerfTimer PT;
  amd_comgr_action_kind_t CurrAction;
  double CurrCounter;
  bool IsValid = false;

public:
  PerfStats() {}
  bool Init(std::string LogFile) {
    std::error_code EC;
    std::unique_ptr<raw_fd_ostream, std::function<void(raw_fd_ostream *)>> LogF(
        new (std::nothrow) raw_fd_ostream(LogFile, EC, sys::fs::OF_Text),
        [](raw_fd_ostream *fp) {
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

    IsValid = true;
    return IsValid;
  }
  void StartAction(amd_comgr_action_kind_t ActionKind) {
    if (!IsValid)
      return;
    CurrAction = ActionKind;
    CurrCounter = PT.getTimer();
  }
  void WriteActionTime() {
    if (!IsValid)
      return;
    Stats[CurrAction].OverallTime += PT.getTimer() - CurrCounter;
    Stats[CurrAction].ExecCounter++;
  }
  void dumpPerfStats() {
    for (uint16_t i = 0; i < AMD_COMGR_ACTION_LAST; i++) {
      if (Stats[i].ExecCounter) {
        *pLog << "Action " << getActionKindName((amd_comgr_action_kind_t)i)
              << " was invoked " << Stats[i].ExecCounter << " times and took "
              << format_decimal(Stats[i].OverallTime, 10)
              << " milliseconds overall\n";
      }
    }
  }
};

}; // namespace TimeStatistics

#endif // AMD_COMGR_TIME_STAT_H
