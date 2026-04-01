//===- aiir-tblgen.cpp - Top-Level TableGen implementation for AIIR -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the main function for AIIR's TableGen IRDL backend.
//
//===----------------------------------------------------------------------===//

#include "aiir/TableGen/GenInfo.h"
#include "aiir/Tools/aiir-tblgen/AiirTblgenMain.h"
#include "llvm/TableGen/Record.h"

using namespace llvm;
using namespace aiir;

// Generator that prints records.
static GenRegistration
    printRecords("print-records", "Print all records to stdout",
                 [](const RecordKeeper &records, raw_ostream &os) {
                   os << records;
                   return false;
                 });

int main(int argc, char **argv) { return AiirTblgenMain(argc, argv); }
