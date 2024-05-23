#ifndef AMD_COMGR_TS_INTERFACE_H
#define AMD_COMGR_TS_INTERFACE_H

#include "llvm/ADT/StringRef.h"
// External interface

namespace COMGR {
namespace TimeStatistics {

struct ProfilePoint {
  ProfilePoint(llvm::StringRef Name);
  ~ProfilePoint();
  void finish();

private:
  std::string Name = "";
  double StartTime = 0.0;
  bool isFinished = false;
};

bool InitTimeStatistics(std::string LogFile);
void StartAction(amd_comgr_action_kind_t);
void EndAction();

} // namespace TimeStatistics
} // namespace COMGR

#endif // AMD_COMGR_TS_INTERFACE_H
