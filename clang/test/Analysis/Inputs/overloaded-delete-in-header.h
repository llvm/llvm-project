#ifndef OVERLOADED_DELETE_IN_HEADER
#define OVERLOADED_DELETE_IN_HEADER

void clang_analyzer_printState();

struct DeleteInHeader {
  inline void operator delete(void *ptr) {
    // No matter whether this header file is included as a system header file
    // with -isystem or a user header file with -I, ptr should not be marked as
    // released.
    clang_analyzer_printState();

    ::operator delete(ptr); // The first place where ptr is marked as released.
  }
};

#endif // OVERLOADED_DELETE_IN_SYSTEM_HEADER
