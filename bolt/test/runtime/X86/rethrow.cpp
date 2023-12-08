#include <iostream>
#include <stdexcept>

void erringFunc() { throw std::runtime_error("Hello"); }

void libCallA() { erringFunc(); }

void libCallB() { throw std::runtime_error("World"); }

void handleEventA() {
  try {
    libCallA();
  } catch (std::runtime_error &E) {
    std::cout << "handleEventA: unhandled error " << E.what() << "\n";
    throw;
  }
}

void handleEventB() {
  try {
    libCallB();
  } catch (std::runtime_error &E) {
    std::cout << "handleEventB: handle error " << E.what() << "\n";
  }
}

class EventGen {
  unsigned RemainingEvents = 5;

public:
  int generateEvent() {
    if (RemainingEvents > 0) {
      --RemainingEvents;
      return (RemainingEvents % 3) + 1;
    }
    return 0;
  }
};

class TerminateException : public std::runtime_error {
public:
  TerminateException() : std::runtime_error("Time to stop!") {}
};

void runEventLoop(EventGen &EG) {
  while (true) {
    try {
      int Ev = EG.generateEvent();
      switch (Ev) {
      case 0:
        throw TerminateException();
      case 1:
        handleEventA();
        break;
      case 2:
        handleEventB();
        break;
      }
    } catch (TerminateException &E) {
      std::cout << "Terminated?\n";
      throw;
    } catch (std::runtime_error &E) {
      std::cout << "Unhandled error: " << E.what() << "\n";
    }
  }
}

struct CleanUp {
  ~CleanUp() { std::cout << "Cleanup\n"; }
};

int main() {
  EventGen EG;
  try {
    CleanUp CU;
    runEventLoop(EG);
  } catch (TerminateException &E) {
    std::cout << "Terminated!\n";
  }
  return 0;
}

/*
REQUIRES: system-linux

RUN: %clang++ %cflags %s -o %t.exe -Wl,-q
RUN: llvm-bolt %t.exe --split-functions --split-strategy=randomN \
RUN:         --split-all-cold --split-eh --bolt-seed=7 -o %t.bolt
RUN: %t.bolt | FileCheck %s

CHECK: handleEventB: handle error World
CHECK-NEXT: handleEventA: unhandled error Hello
CHECK-NEXT: Unhandled error: Hello
CHECK-NEXT: handleEventB: handle error World
CHECK-NEXT: handleEventA: unhandled error Hello
CHECK-NEXT: Unhandled error: Hello
CHECK-NEXT: Terminated?
CHECK-NEXT: Cleanup
CHECK-NEXT: Terminated!
*/
