// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir -std=c++17
// XFAIL: *
//
// std::filesystem with range-for crashes - SD_Automatic not implemented
// Location: CIRGenExpr.cpp:2356

#include <filesystem>

void test() {
  namespace fs = std::filesystem;

  // This triggers SD_Automatic for the directory_iterator temporary
  for (const auto& entry : fs::directory_iterator("/tmp")) {
    auto path = entry.path();
  }
}
