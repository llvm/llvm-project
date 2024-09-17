/*
Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef SRC_HIPBIN_UTIL_H_
#define SRC_HIPBIN_UTIL_H_

#include "filesystem.h"

#include "utils.h"

#include <assert.h>
#include <stdlib.h>
#include <limits.h>
#include <iostream>
#include <sstream>
#include <string>
#include <map>
#include <fstream>
#include <regex>
#include <algorithm>
#include <vector>


#if defined(_WIN32) || defined(_WIN64)
#include <tchar.h>
#include <windows.h>
#include <io.h>
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

using std::cout;
using std::endl;
using std::vector;
using std::string;
using std::ifstream;
using std::ofstream;
using std::regex;
using std::regex_match;
using std::regex_search;
using std::regex_replace;
using std::map;
using std::smatch;
using std::stringstream;


struct SystemCmdOut {
  string out;
  int exitCode = 0;
};

class HipBinUtil {
 public:
  static HipBinUtil* getInstance() {
      if (!instance)
      instance = new HipBinUtil;
      return instance;
  }
  virtual ~HipBinUtil();
  // Common helper functions
  vector<string> splitStr(string fullStr, char delimiter) const;
  string replaceStr(const string& s, const string& toReplace,
                    const string& replaceWith) const;
  string replaceRegex(const string& s, regex toReplace,
                      string replaceWith) const;
  SystemCmdOut exec(const char* cmd, bool printConsole) const;
  string getTempDir();
  void deleteTempFiles();
  string mktempFile(string name);
  string trim(string str) const;
  string readConfigMap(map<string, string> hipVersionMap,
                       string keyName, string defaultValue) const;
  map<string, string> parseConfigFile(fs::path configPath) const;
  bool substringPresent(string fullString, string subString) const;
  bool stringRegexMatch(string fullString, string pattern) const;
  bool checkCmd(const vector<string>& commands, const string& argument);

 private:
  HipBinUtil() {}
  vector<string> tmpFiles_;
  static HipBinUtil *instance;
};

HipBinUtil *HipBinUtil::instance = 0;

// deleting temp files created
HipBinUtil::~HipBinUtil() {
  deleteTempFiles();
}

// create temp file with the template name
string HipBinUtil::mktempFile(string name) {
  string fileName;
#if defined(_WIN32) || defined(_WIN64)
  fileName = _mktemp(&name[0]);
#else
  fileName = mktemp(&name[0]);
#endif
  tmpFiles_.push_back(fileName);
  return fileName;
}

// removes the empty spaces and end lines
string HipBinUtil::trim(string str) const {
  string strChomp = str;
  strChomp.erase(str.find_last_not_of(" \n\r\t")+1);
  return strChomp;
}

// matches the pattern in the string
bool HipBinUtil::stringRegexMatch(string fullString, string pattern) const {
  return regex_match(fullString, regex(pattern));
}

// subtring is present in string
bool HipBinUtil::substringPresent(string fullString, string subString) const {
  return fullString.find(subString) != string::npos;
}

// splits the string with the delimiter
vector<string> HipBinUtil::splitStr(string fullStr, char delimiter) const {
  vector <string> tokens;
  stringstream check1(fullStr);
  string intermediate;
  while (getline(check1, intermediate, delimiter)) {
    tokens.push_back(intermediate);
  }
  return tokens;
}

// replaces the toReplace string with replaceWith string. Returns the new string
string HipBinUtil::replaceStr(const string& s, const string& toReplace,
                              const string& replaceWith) const {
  string out = s;
  std::size_t pos = out.find(toReplace);
  if (pos == string::npos) return out;
  return out.replace(pos, toReplace.length(), replaceWith);
}

// replaces the toReplace regex pattern with replaceWith string.
// Returns the new string
string HipBinUtil::replaceRegex(const string& s, regex toReplace,
                                string replaceWith) const {
  string out = s;
  while (regex_search(out, toReplace)) {
    out = regex_replace(out, toReplace, replaceWith);
  }
  return out;
}

// reads the config file and stores it in a map for access
map<string, string> HipBinUtil::parseConfigFile(fs::path configPath) const {
  map<string, string> configMap;
  ifstream isFile(configPath.string());
  string line;
  if (isFile.is_open()) {
    while (std::getline(isFile, line)) {
      std::istringstream is_line(line);
      string key;
      if (std::getline(is_line, key, '=')) {
        string value;
        if (std::getline(is_line, value)) {
          configMap.insert({ key, value });
        }
      }
    }
    isFile.close();
  }
  return configMap;
}

// Delete all created temporary files
void HipBinUtil::deleteTempFiles() {
  // Deleting temp files vs the temp directory
  for (unsigned int i = 0; i < tmpFiles_.size(); i++) {
    try {
      if (!fs::remove(tmpFiles_.at(i)))
        std::cerr << "Error deleting temp name: "<< tmpFiles_.at(i) <<endl;
    }
    catch(...) {
      std::cerr << "Error deleting temp name: "<< tmpFiles_.at(i) <<endl;
    }
  }
}

// Create a new temporary directory and return it
string HipBinUtil::getTempDir() {
  // mkdtemp is only applicable for unix and not windows.
  // Using filesystem becasuse of windows limitation
  string tmpdir = fs::temp_directory_path().string();
  // tmpDirs_.push_back(tmpdir);
  return tmpdir;
}

// executes the command, returns the status and return string
SystemCmdOut HipBinUtil::exec(const char* cmd,
                              bool printConsole = false) const {
  SystemCmdOut sysOut;
  try {
    char buffer[128];
    string result = "";
    #if defined(_WIN32) || defined(_WIN64)
      FILE* pipe = _popen(cmd, "r");
    #else
      FILE* pipe = popen(cmd, "r");
    #endif
    if (!pipe) throw std::runtime_error("popen() failed!");
    try {
      while (fgets(buffer, sizeof buffer, pipe) != NULL) {
        result += buffer;
      }
    } catch (...) {
      std::cerr << "Error while executing the command: " << cmd << endl;
    }
    #if defined(_WIN32) || defined(_WIN64)
      sysOut.exitCode = _pclose(pipe);
    #else
      int closeStatus = pclose(pipe);
      sysOut.exitCode =  WEXITSTATUS(closeStatus);
    #endif
    if (printConsole == true) {
      cout << result;
    }
    sysOut.out = result;
  }
  catch(...) {
    sysOut.exitCode = -1;
  }
  return sysOut;
}

// returns the value of the key from the Map passed
string HipBinUtil::readConfigMap(map<string, string> hipVersionMap,
                                 string keyName, string defaultValue) const {
  auto it = hipVersionMap.find(keyName);
  if (it != hipVersionMap.end()) {
    return it->second;
  }
  return defaultValue;
}



bool HipBinUtil::checkCmd(const vector<string>& commands,
                          const string& argument) {
  bool found = false;
  for (unsigned int i = 0; i < commands.size(); i++) {
    if (argument.compare(commands.at(i)) == 0) {
      found = true;
      break;
    }
  }
  return found;
}



#endif  // SRC_HIPBIN_UTIL_H_
