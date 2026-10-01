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
    "8": "32bit",
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


def csv2def(csv_path, def_path):
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
