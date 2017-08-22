// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.

// Parse some flags
#include <string>
#include <vector>

static std::vector<std::string> Flags;

extern "C" int LLVMFuzzerInitialize(int *Argc, char ***Argv) {
  // Parse --flags and anything after -ignore_remaining_args=1 is passed.
  int I = 1;
  while (I < *Argc) {
    std::string S((*Argv)[I++]);
    if (S == "-ignore_remaining_args=1")
      break;
    if (S.substr(0, 2) == "--")
      Flags.push_back(S);
  }
  while (I < *Argc)
    Flags.push_back(std::string((*Argv)[I++]));

  return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  fprintf(stderr, "BINGO ");
  for (auto Flag : Flags)
    fprintf(stderr, "%s ", Flag.c_str());
  fprintf(stderr, "\n");
  exit(0);
}
