// REQUIRES: x86-registered-target
// RUN: %clangxx -c %s -o %t_fat.o
// RUN: %clangxx %t_fat.o -o %t.exe
// RUN: clang-offload-bundler -type=oo -targets=host-x86_64-unknown-linux-gnu,sycl-x86_64-unknown-linux-gnu -outputs=%t.o,%t_list.txt -inputs=%t_fat.o -unbundle
// RUN: %t.exe %t_list.txt | FileCheck %s
// CHECK:11
// CHECK:222
// CHECK:3333

#include <cstdint>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
using namespace std;

#define BUNDLE_SECTION_PREFIX "__CLANG_OFFLOAD_BUNDLE__"
#define BUNDLE_SIZE_SECTION_PREFIX "__CLANG_OFFLOAD_BUNDLE_SIZE__"

#define TARGET0 "host-x86_64-unknown-linux-gnu"
#define TARGET1 "sycl-x86_64-unknown-linux-gnu"

// Populate section with special names recognized by the bundler;
// this emulates fat object partially linked from 3 other fat objects.
// The test uses the bundler to split the bundle into 3 objects and then prints
// their contents to stdout.
char str0[] __attribute__((section(BUNDLE_SECTION_PREFIX TARGET0))) = { 0, 0, 0 };
int64_t size0[] __attribute__((section(BUNDLE_SIZE_SECTION_PREFIX TARGET0))) = { 1, 1, 1 };

char str1[] __attribute__((section(BUNDLE_SECTION_PREFIX TARGET1))) = { "11\n222\n3333\n" };
int64_t size1[] __attribute__((section(BUNDLE_SIZE_SECTION_PREFIX TARGET1))) = { 3, 4, 5 };

void cat(const string& File) {
  string Line;
  ifstream F(File);
  if (F.is_open()) {
    while (getline(F, Line)) {
      cout << Line << '\n';
    }
    F.close();
  }
  else cout << "Unable to open file " << File;
}

// main is invoked with the bundler output file as argument -
// read this file and print their contents to stdout.
int main(int argc, char **argv) {
  string ListFile(argv[1]);
  string Line;
  ifstream F(ListFile);
  vector<string> OutFiles;

  if (F.is_open()) {
    while (getline(F, Line)) {
      OutFiles.push_back(Line);
    }
    F.close();
  }
  else {
    cout << "Unable to open file " << ListFile;
    return 1;
  }

  for (const auto &File : OutFiles) {
    cat(File);
  }
  return 0;
}

