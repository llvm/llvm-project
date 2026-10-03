// RUN: %clangxx %cxxflags -O1 -Wl,-q %s -o %t.exe
// RUN: %t.exe

/// Splitting spreads the EH ranges of classify() over several fragments. Each
/// fragment gets its own LSDA, but they all share a single type table, reached
/// through the @TType base offset in their headers. Selecting the right catch
/// clause therefore exercises the shared table from more than one LSDA.
// RUN: llvm-bolt %t.exe -o %t.bolt --split-functions --split-strategy=all \
// RUN:   --split-eh
// RUN: %t.bolt

// REQUIRES: system-linux

#include <cstdio>
#include <cstring>
#include <stdexcept>

__attribute__((noinline)) void thrower(int Kind) {
  switch (Kind) {
  case 0:
    throw std::out_of_range("oor");
  case 1:
    throw std::invalid_argument("ia");
  default:
    throw std::overflow_error("of");
  }
}

/// Several catch clauses, hence several type table entries. Picking the right
/// one depends on the indices resolving against the shared table.
__attribute__((noinline)) const char *classify(int Kind) {
  try {
    thrower(Kind);
  } catch (const std::out_of_range &) {
    return "out_of_range";
  } catch (const std::invalid_argument &) {
    return "invalid_argument";
  } catch (const std::overflow_error &) {
    return "overflow_error";
  } catch (...) {
    return "other";
  }
  return "none";
}

int main() {
  static const char *const Expected[] = {"out_of_range", "invalid_argument",
                                         "overflow_error"};
  for (int Kind = 0; Kind < 3; ++Kind) {
    const char *Result = classify(Kind);
    std::printf("%d caught as: %s\n", Kind, Result);
    if (std::strcmp(Result, Expected[Kind]) != 0)
      return 1;
  }
  return 0;
}
