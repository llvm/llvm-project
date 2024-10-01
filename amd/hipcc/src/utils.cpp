#include "utils.h"
#include "filesystem.h"

#if defined(_WIN32) || defined(_WIN64)
#include <io.h>
#include <tchar.h>
#include <windows.h>
#ifdef _UNICODE
typedef wchar_t TCHAR;
typedef std::wstring TSTR;
typedef std::wstring::size_type TSIZE;
#define ENDLINE L"/\\"
#else
typedef char TCHAR;
typedef std::string TSTR;
typedef std::string::size_type TSIZE;
#define ENDLINE "/\\"
#endif
#else
#include <unistd.h>
#endif

#include <iostream>
#include <sstream>

std::string hipcc::utils::getSelfPath() {
  constexpr size_t MAX_PATH_CHAR = 1024;
  std::string path;
#if defined(_WIN32) || defined(_WIN64)
  TCHAR buffer[MAX_PATH] = {0};
  GetModuleFileName(NULL, buffer, MAX_PATH_CHAR);
  TSIZE pos = TSTR(buffer).find_last_of(ENDLINE);
  TSTR wide = TSTR(buffer).substr(0, pos);
  path = std::string(wide.begin(), wide.end());
#else
  char buff[MAX_PATH_CHAR];
  ssize_t len = ::readlink("/proc/self/exe", buff, sizeof(buff) - 1);
  if (len > 0) {
    buff[len] = '\0';
    path = std::string(buff);
    fs::path exePath(path);
    path = exePath.parent_path().string();
  } else {
    std::cerr << "readlink: Error reading the exe path" << std::endl;
    perror("readlink");
    exit(-1);
  }
#endif
  return path;
}

std::vector<std::string> hipcc::utils::splitStr(std::string const &fullStr,
                                                char delimiter) {
  std::vector<std::string> tokens;
  std::stringstream check1(fullStr);
  std::string intermediate;
  while (std::getline(check1, intermediate, delimiter)) {
    tokens.emplace_back(std::move(intermediate));
  }
  return tokens;
}