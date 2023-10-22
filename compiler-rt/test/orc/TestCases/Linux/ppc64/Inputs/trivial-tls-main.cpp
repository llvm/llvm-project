#include <fstream>
#include <iostream>
#include <string>

thread_local int x = 0;
thread_local int y = 1;
thread_local int z = -1;

extern int TestPOWER10();

int Test() { return x + y + z; }

static bool CPUModelIsPOWER10() {
  std::string line;
  std::ifstream cpuinfo("/proc/cpuinfo", std::ios::in);
  if (!cpuinfo.is_open())
    return false;
  while (std::getline(cpuinfo, line)) {
    if (line.find("cpu") != std::string::npos &&
        line.find("POWER10") != std::string::npos)
      return true;
  }
  return false;
}

int main() {
  if (CPUModelIsPOWER10())
    return TestPOWER10();
  return Test();
}
