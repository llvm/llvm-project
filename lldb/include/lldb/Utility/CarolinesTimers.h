// This is part of a temporary patch, to collect start-up time pieces in LLDB.

#ifndef LLDB_CAROLINES_TIMERS_H
#define LLDB_CAROLINES_TIMERS_H

#include "lldb/lldb-defines.h"
#include "llvm/Support/Chrono.h"
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <string>
#include <vector>

namespace lldb_private {

enum CarolineTimerEvent {
  eCarolineStartFrameVar = 0,
  eCarolineEndFrameVar = 1,
  eCarolineStartExprEval = 2,
  eCarolineEndExprEval = 3,
  //eCarolineStartNameIndexes = 4,
  //eCarolineEndNameIndexes = 5,
  //eCarolineStartDwarfIndex = 6,
  //eCarolineEndDwarfIndex = 7,
  //eCarolineEndHandleCommand = 8,
  eCarolineLastItem = 4
};

void CarolineTimeStamp(CarolineTimerEvent event_kind,
                       const std::string &expr,
                       timespec *time_ptr = nullptr);


} // namespace lldb_private

#endif // LLDB_CAROLINES_TIMERS_H
