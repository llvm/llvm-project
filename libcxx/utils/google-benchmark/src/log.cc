#include "log.h"

#include <iostream>

namespace benchmark {
namespace internal {

int& LoggingLevelImp() {
    static int level = 0;
    return level;
}

void SetLogLevel(int value) {
    LoggingLevelImp() = value;
}

int GetLogLevel() {
    return LoggingLevelImp();
}

class NullLogBuffer : public std::streambuf
{
public:
  int overflow(int c) {
    return c;
  }
};

std::ostream& GetNullLogInstance() {
  static NullLogBuffer log_buff;
  static std::ostream null_log(&log_buff);
  return null_log;
}

std::ostream& GetErrorLogInstance() {
  return std::clog;
}

} // end namespace internal
} // end namespace benchmark