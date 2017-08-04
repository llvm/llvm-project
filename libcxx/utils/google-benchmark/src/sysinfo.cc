// Copyright 2015 Google Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "sysinfo.h"
#include "internal_macros.h"

#ifdef BENCHMARK_OS_WINDOWS
#include <Shlwapi.h>
#include <VersionHelpers.h>
#include <Windows.h>
#else
#include <fcntl.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <sys/types.h>  // this header must be included before 'sys/sysctl.h' to avoid compilation error on FreeBSD
#include <unistd.h>
#if defined BENCHMARK_OS_FREEBSD || defined BENCHMARK_OS_MACOSX
#include <sys/sysctl.h>
#endif
#endif

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <mutex>

#include "arraysize.h"
#include "check.h"
#include "cycleclock.h"
#include "internal_macros.h"
#include "log.h"
#include "sleep.h"
#include "string_util.h"

namespace benchmark {
namespace {
std::once_flag cpuinfo_init;
double cpuinfo_cycles_per_second = 1.0;
int cpuinfo_num_cpus = 1;  // Conservative guess

#if !defined BENCHMARK_OS_MACOSX
const int64_t estimate_time_ms = 1000;

// Helper function estimates cycles/sec by observing cycles elapsed during
// sleep(). Using small sleep time decreases accuracy significantly.
int64_t EstimateCyclesPerSecond() {
  const int64_t start_ticks = cycleclock::Now();
  SleepForMilliseconds(estimate_time_ms);
  return cycleclock::Now() - start_ticks;
}
#endif

#if defined BENCHMARK_OS_LINUX || defined BENCHMARK_OS_CYGWIN
// Helper function for reading an int from a file. Returns true if successful
// and the memory location pointed to by value is set to the value read.
bool ReadIntFromFile(const char* file, long* value) {
  bool ret = false;
  int fd = open(file, O_RDONLY);
  if (fd != -1) {
    char line[1024];
    char* err;
    memset(line, '\0', sizeof(line));
    ssize_t read_err = read(fd, line, sizeof(line) - 1);
    ((void)read_err); // prevent unused warning
    CHECK(read_err >= 0);
    const long temp_value = strtol(line, &err, 10);
    if (line[0] != '\0' && (*err == '\n' || *err == '\0')) {
      *value = temp_value;
      ret = true;
    }
    close(fd);
  }
  return ret;
}
#endif

#if defined BENCHMARK_OS_LINUX || defined BENCHMARK_OS_CYGWIN
static std::string convertToLowerCase(std::string s) {
  for (auto& ch : s)
    ch = std::tolower(ch);
  return s;
}
static bool startsWithKey(std::string Value, std::string Key,
                          bool IgnoreCase = true) {
  if (IgnoreCase) {
    Key = convertToLowerCase(std::move(Key));
    Value = convertToLowerCase(std::move(Value));
  }
  return Value.compare(0, Key.size(), Key) == 0;
}
#endif

void InitializeSystemInfo() {
#if defined BENCHMARK_OS_LINUX || defined BENCHMARK_OS_CYGWIN
  char line[1024];
  char* err;
  long freq;

  bool saw_mhz = false;

  // If the kernel is exporting the tsc frequency use that. There are issues
  // where cpuinfo_max_freq cannot be relied on because the BIOS may be
  // exporintg an invalid p-state (on x86) or p-states may be used to put the
  // processor in a new mode (turbo mode). Essentially, those frequencies
  // cannot always be relied upon. The same reasons apply to /proc/cpuinfo as
  // well.
  if (!saw_mhz &&
      ReadIntFromFile("/sys/devices/system/cpu/cpu0/tsc_freq_khz", &freq)) {
    // The value is in kHz (as the file name suggests).  For example, on a
    // 2GHz warpstation, the file contains the value "2000000".
    cpuinfo_cycles_per_second = freq * 1000.0;
    saw_mhz = true;
  }

  // If CPU scaling is in effect, we want to use the *maximum* frequency,
  // not whatever CPU speed some random processor happens to be using now.
  if (!saw_mhz &&
      ReadIntFromFile("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq",
                      &freq)) {
    // The value is in kHz.  For example, on a 2GHz warpstation, the file
    // contains the value "2000000".
    cpuinfo_cycles_per_second = freq * 1000.0;
    saw_mhz = true;
  }

  // Read /proc/cpuinfo for other values, and if there is no cpuinfo_max_freq.
  const char* pname = "/proc/cpuinfo";
  int fd = open(pname, O_RDONLY);
  if (fd == -1) {
    perror(pname);
    if (!saw_mhz) {
      cpuinfo_cycles_per_second =
          static_cast<double>(EstimateCyclesPerSecond());
    }
    return;
  }

  double bogo_clock = 1.0;
  bool saw_bogo = false;
  long max_cpu_id = 0;
  int num_cpus = 0;
  line[0] = line[1] = '\0';
  size_t chars_read = 0;
  do {  // we'll exit when the last read didn't read anything
    // Move the next line to the beginning of the buffer
    const size_t oldlinelen = strlen(line);
    if (sizeof(line) == oldlinelen + 1)  // oldlinelen took up entire line
      line[0] = '\0';
    else  // still other lines left to save
      memmove(line, line + oldlinelen + 1, sizeof(line) - (oldlinelen + 1));
    // Terminate the new line, reading more if we can't find the newline
    char* newline = strchr(line, '\n');
    if (newline == nullptr) {
      const size_t linelen = strlen(line);
      const size_t bytes_to_read = sizeof(line) - 1 - linelen;
      CHECK(bytes_to_read > 0);  // because the memmove recovered >=1 bytes
      chars_read = read(fd, line + linelen, bytes_to_read);
      line[linelen + chars_read] = '\0';
      newline = strchr(line, '\n');
    }
    if (newline != nullptr) *newline = '\0';

    // When parsing the "cpu MHz" and "bogomips" (fallback) entries, we only
    // accept postive values. Some environments (virtual machines) report zero,
    // which would cause infinite looping in WallTime_Init.
    if (!saw_mhz && startsWithKey(line, "cpu MHz")) {
      const char* freqstr = strchr(line, ':');
      if (freqstr) {
        cpuinfo_cycles_per_second = strtod(freqstr + 1, &err) * 1000000.0;
        if (freqstr[1] != '\0' && *err == '\0' && cpuinfo_cycles_per_second > 0)
          saw_mhz = true;
      }
    } else if (startsWithKey(line, "bogomips")) {
      const char* freqstr = strchr(line, ':');
      if (freqstr) {
        bogo_clock = strtod(freqstr + 1, &err) * 1000000.0;
        if (freqstr[1] != '\0' && *err == '\0' && bogo_clock > 0)
          saw_bogo = true;
      }
    } else if (startsWithKey(line, "processor", /*IgnoreCase*/false)) {
      // The above comparison is case-sensitive because ARM kernels often
      // include a "Processor" line that tells you about the CPU, distinct
      // from the usual "processor" lines that give you CPU ids. No current
      // Linux architecture is using "Processor" for CPU ids.
      num_cpus++;  // count up every time we see an "processor :" entry
      const char* id_str = strchr(line, ':');
      if (id_str) {
        const long cpu_id = strtol(id_str + 1, &err, 10);
        if (id_str[1] != '\0' && *err == '\0' && max_cpu_id < cpu_id)
          max_cpu_id = cpu_id;
      }
    }
  } while (chars_read > 0);
  close(fd);

  if (!saw_mhz) {
    if (saw_bogo) {
      // If we didn't find anything better, we'll use bogomips, but
      // we're not happy about it.
      cpuinfo_cycles_per_second = bogo_clock;
    } else {
      // If we don't even have bogomips, we'll use the slow estimation.
      cpuinfo_cycles_per_second =
          static_cast<double>(EstimateCyclesPerSecond());
    }
  }
  if (num_cpus == 0) {
    fprintf(stderr, "Failed to read num. CPUs correctly from /proc/cpuinfo\n");
  } else {
    if ((max_cpu_id + 1) != num_cpus) {
      fprintf(stderr,
              "CPU ID assignments in /proc/cpuinfo seem messed up."
              " This is usually caused by a bad BIOS.\n");
    }
    cpuinfo_num_cpus = num_cpus;
  }

#elif defined BENCHMARK_OS_FREEBSD
// For this sysctl to work, the machine must be configured without
// SMP, APIC, or APM support.  hz should be 64-bit in freebsd 7.0
// and later.  Before that, it's a 32-bit quantity (and gives the
// wrong answer on machines faster than 2^32 Hz).  See
//  http://lists.freebsd.org/pipermail/freebsd-i386/2004-November/001846.html
// But also compare FreeBSD 7.0:
//  http://fxr.watson.org/fxr/source/i386/i386/tsc.c?v=RELENG70#L223
//  231         error = sysctl_handle_quad(oidp, &freq, 0, req);
// To FreeBSD 6.3 (it's the same in 6-STABLE):
//  http://fxr.watson.org/fxr/source/i386/i386/tsc.c?v=RELENG6#L131
//  139         error = sysctl_handle_int(oidp, &freq, sizeof(freq), req);
#if __FreeBSD__ >= 7
  uint64_t hz = 0;
#else
  unsigned int hz = 0;
#endif
  size_t sz = sizeof(hz);
  const char* sysctl_path = "machdep.tsc_freq";
  if (sysctlbyname(sysctl_path, &hz, &sz, nullptr, 0) != 0) {
    fprintf(stderr, "Unable to determine clock rate from sysctl: %s: %s\n",
            sysctl_path, strerror(errno));
    cpuinfo_cycles_per_second = static_cast<double>(EstimateCyclesPerSecond());
  } else {
    cpuinfo_cycles_per_second = hz;
  }
// TODO: also figure out cpuinfo_num_cpus

#elif defined BENCHMARK_OS_WINDOWS
  // In NT, read MHz from the registry. If we fail to do so or we're in win9x
  // then make a crude estimate.
  DWORD data, data_size = sizeof(data);
  if (IsWindowsXPOrGreater() &&
      SUCCEEDED(
          SHGetValueA(HKEY_LOCAL_MACHINE,
                      "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
                      "~MHz", nullptr, &data, &data_size)))
    cpuinfo_cycles_per_second =
        static_cast<double>((int64_t)data * (int64_t)(1000 * 1000));  // was mhz
  else
    cpuinfo_cycles_per_second = static_cast<double>(EstimateCyclesPerSecond());

  SYSTEM_INFO sysinfo;
  // Use memset as opposed to = {} to avoid GCC missing initializer false
  // positives.
  std::memset(&sysinfo, 0, sizeof(SYSTEM_INFO));
  GetSystemInfo(&sysinfo);
  cpuinfo_num_cpus = sysinfo.dwNumberOfProcessors;  // number of logical
                                                    // processors in the current
                                                    // group

#elif defined BENCHMARK_OS_MACOSX
  int32_t num_cpus = 0;
  size_t size = sizeof(num_cpus);
  if (::sysctlbyname("hw.ncpu", &num_cpus, &size, nullptr, 0) == 0 &&
      (size == sizeof(num_cpus))) {
    cpuinfo_num_cpus = num_cpus;
  } else {
    fprintf(stderr, "%s\n", strerror(errno));
    std::exit(EXIT_FAILURE);
  }
  int64_t cpu_freq = 0;
  size = sizeof(cpu_freq);
  if (::sysctlbyname("hw.cpufrequency", &cpu_freq, &size, nullptr, 0) == 0 &&
      (size == sizeof(cpu_freq))) {
    cpuinfo_cycles_per_second = cpu_freq;
  } else {
    #if defined BENCHMARK_OS_IOS
    fprintf(stderr, "CPU frequency cannot be detected. \n");
    cpuinfo_cycles_per_second = 0;
    #else
    fprintf(stderr, "%s\n", strerror(errno));
    std::exit(EXIT_FAILURE);
    #endif
  }
#else
  // Generic cycles per second counter
  cpuinfo_cycles_per_second = static_cast<double>(EstimateCyclesPerSecond());
#endif
}

}  // end namespace

double CyclesPerSecond(void) {
  std::call_once(cpuinfo_init, InitializeSystemInfo);
  return cpuinfo_cycles_per_second;
}

int NumCPUs(void) {
  std::call_once(cpuinfo_init, InitializeSystemInfo);
  return cpuinfo_num_cpus;
}

// The ""'s catch people who don't pass in a literal for "str"
#define strliterallen(str) (sizeof("" str "") - 1)

// Must use a string literal for prefix.
#define memprefix(str, len, prefix)                       \
  ((((len) >= strliterallen(prefix)) &&                   \
    std::memcmp(str, prefix, strliterallen(prefix)) == 0) \
       ? str + strliterallen(prefix)                      \
       : nullptr)

bool CpuScalingEnabled() {
#ifndef BENCHMARK_OS_WINDOWS
  // On Linux, the CPUfreq subsystem exposes CPU information as files on the
  // local file system. If reading the exported files fails, then we may not be
  // running on Linux, so we silently ignore all the read errors.
  for (int cpu = 0, num_cpus = NumCPUs(); cpu < num_cpus; ++cpu) {
    std::string governor_file =
        StrCat("/sys/devices/system/cpu/cpu", cpu, "/cpufreq/scaling_governor");
    FILE* file = fopen(governor_file.c_str(), "r");
    if (!file) break;
    char buff[16];
    size_t bytes_read = fread(buff, 1, sizeof(buff), file);
    fclose(file);
    if (memprefix(buff, bytes_read, "performance") == nullptr) return true;
  }
#endif
  return false;
}

}  // end namespace benchmark
