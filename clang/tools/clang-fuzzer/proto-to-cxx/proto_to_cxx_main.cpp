//==-- proto_to_cxx_main.cpp - Driver for protobuf-C++ conversion ----------==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Implements a simple driver to print a C++ program from a protobuf.
//
//===----------------------------------------------------------------------===//
#include <fstream>
#include <iostream>
#include <streambuf>
#include <string>

#include "proto_to_cxx.h"

int main(int argc, char **argv) {
  for (int i = 1; i < argc; i++) {
    std::fstream in(argv[i]);
    std::string str((std::istreambuf_iterator<char>(in)),
                    std::istreambuf_iterator<char>());
    std::cout << "// " << argv[i] << std::endl;
    std::cout << clang_fuzzer::ProtoToCxx(
        reinterpret_cast<const uint8_t *>(str.data()), str.size());
  }
}

