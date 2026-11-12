// RUN: rm -rf %t
// RUN: %clang_cc1 -std=c++11 -internal-externc-isystem %S/Inputs/PR27739 -verify %s
// RUN: %clang_cc1 -std=c++11 -fmodules -fmodule-map-file=%S/Inputs/PR27739/module.modulemap -fmodules-cache-path=%t -internal-externc-isystem %S/Inputs/PR27739/ -verify %s

#include "DataInputHandler.h"

void DataInputHandler::AddTree() {
   fInputTrees[(char*)""];
   fExplicitTrainTest[(char*)""];
}

// expected-no-diagnostics
