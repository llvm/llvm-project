#!/usr/bin/env python3
# ===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===##
"""
Script to generate MSP430 definitions from TI's devices.csv

Download the devices.csv from [1] using the link "Header and Support Files".

CPUS and MULTIPLIERS map the numeric CPU_TYPE and MPY_TYPE codes of the CSV
to the CPU and hwmult names used by clang. The codes are defined in the
legend at the top of devices.csv:

CPU_TYPE:
  0: MSP430 CPU
  1: MSP430X CPU
  2: MSP430XV2 CPU

MPY_TYPE:
  0: No hardware multiplier
  1: 16 bit hardware multiplier
  2: 16 bit hardware multiplier with sign extension (2xx devices)
  4: 32 bit hardware multiplier
  8: 32 bit hardware multiplier (5xx devices)

Code 2 maps to "16bit" because there is no separate hwmult option for it.
Code 8 maps to "f5series" because the 5xx/6xx multiplier uses different
register addresses than the other 32 bit multiplier.

[1]: https://www.ti.com/tool/MSP430-GCC-OPENSOURCE#downloads
"""
import csv
import sys

DEVICE_COLUMN = 0
CPU_COLUMN = 1
MULTIPLIER_COLUMN = 3

CPUS = {"0": "msp430", "1": "msp430x", "2": "msp430xv2"}
MULTIPLIERS = {
    "0": "none",
    "1": "16bit",
    "2": "16bit",
    "4": "32bit",
    "8": "f5series",
}

PREFIX = """//===--- MSP430Target.def - MSP430 Feature/Processor Database----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the MSP430 devices and their features.
//
// Generated from TI's devices.csv in version {} using the script in
// Target/MSP430/gen-msp430-def.py - use this tool rather than adding
// new MCUs by hand.
//
// You can download the devices.csv from https://www.ti.com/tool/MSP430-GCC-OPENSOURCE#overview
// by downloading the support and header files.
//
// This script exports the "Device Name", "CPU_TYPE" and "MPY_TYPE" columns from the
// devices script.
//
// CPU_TYPE codes:
//   0: MSP430 CPU          -> "msp430"
//   1: MSP430X CPU         -> "msp430x"
//   2: MSP430XV2 CPU       -> "msp430xv2"
//
// MPY_TYPE codes:
//   0: No hardware multiplier                         -> "none"
//   1: 16 bit hardware multiplier                     -> "16bit"
//   2: 16 bit hardware multiplier with sign extension -> "16bit"
//   4: 32 bit hardware multiplier                     -> "32bit"
//   8: 32 bit hardware multiplier (5xx devices)       -> "f5series"
//
// Code 2 maps to "16bit" because there is no separate hwmult option for it.
// Code 8 maps to "f5series" because the 5xx/6xx multiplier uses different
// register addresses than the other 32 bit multiplier.
//===----------------------------------------------------------------------===//

#ifndef MSP430_MCU
#define MSP430_MCU(NAME, CPU, HWMULT)
#endif

"""

SUFFIX = """
// Generic MCUs
MSP430_MCU("msp430i2xxgeneric", "msp430", "none")

#undef MSP430_MCU
"""


def csv2def(csv_path: str, def_path: str):
    """
    Parse the devices.csv file at the given path, generate the definitions and
    write them to the given path.

    :param csv_path: Path to the devices.csv to parse
    :type csv_path: str
    :param def_path: Path to the output file to write the definitions to
    "type def_path: str
    """

    mcus = []
    version = "unknown"

    with open(csv_path) as csv_file:
        csv_reader = csv.reader(csv_file)
        while True:
            row = next(csv_reader)
            if len(row) < MULTIPLIER_COLUMN:
                continue

            if row[DEVICE_COLUMN] == "# Device Name":
                assert row[CPU_COLUMN] == "CPU_TYPE", "File format changed"
                assert row[MULTIPLIER_COLUMN] == "MPY_TYPE", "File format changed"
                break

            if row[0] == "Version:":
                version = row[1]

        for row in csv_reader:
            if row[DEVICE_COLUMN].endswith("generic"):
                continue
            assert row[CPU_COLUMN] in CPUS, "Unknown CPU type"
            assert row[MULTIPLIER_COLUMN] in MULTIPLIERS, "Unknown multiplier type"
            mcus.append(
                (
                    row[DEVICE_COLUMN],
                    CPUS[row[CPU_COLUMN]],
                    MULTIPLIERS[row[MULTIPLIER_COLUMN]],
                )
            )

    with open(def_path, "w") as def_file:
        def_file.write(PREFIX.format(version))

        for name, cpu, hwmult in mcus:
            def_file.write(f'MSP430_MCU("{name}", "{cpu}", "{hwmult}")\n')

        def_file.write(SUFFIX)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(f"Usage: {sys.argv[0]} <CSV_FILE> <DEF_FILE>")

    csv2def(sys.argv[1], sys.argv[2])
