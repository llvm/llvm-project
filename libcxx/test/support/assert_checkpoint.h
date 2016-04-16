#ifndef SUPPORT_ASSERT_CHECKPOINT_H
#define SUPPORT_ASSERT_CHECKPOINT_H

#include <csignal>
#include <iostream>
#include <cstdlib>

struct Checkpoint {
  const char* file;
  const char* func;
  int line;
  const char* msg;

  template <class Stream>
  void print(Stream& s) const {
      if (!file) {
          s << "NO CHECKPOINT\n";
          return;
      }
      s << file << ":" << line << " " << func << ": Checkpoint";
      if (msg)
        s << " '" << msg << "'";
      s << std::endl;
  }
};

inline Checkpoint& globalCheckpoint() {
    static Checkpoint C;
    return C;
}

inline void clearCheckpoint() {
    globalCheckpoint() = Checkpoint{0};
}

#define CHECKPOINT(msg) globalCheckpoint() = Checkpoint{__FILE__, __PRETTY_FUNCTION__, __LINE__, msg}

inline void checkpointSignalHandler(int signal) {
    if (signal == SIGABRT) {
        globalCheckpoint().print(std::cerr);
    } else {
        std::cerr << "Unexpected signal " << signal << " received\n";
    }
    std::_Exit(EXIT_FAILURE);
}

inline bool initCheckpointHandler() {
    typedef void(*HandlerT)(int);
    static bool isInit = false;
    if (isInit) return true;
    HandlerT prev_h = std::signal(SIGABRT, checkpointSignalHandler);
    if (prev_h == SIG_ERR) {
        std::cerr << "Setup failed.\n";
        std::_Exit(EXIT_FAILURE);
    }
    isInit = true;
    return false;
}

static bool initDummy = initCheckpointHandler();

#endif
