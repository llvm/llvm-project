#!/usr/bin/env python
#
# ===- csv2json.py - Static Analyzer test helper ---*- python -*-===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===------------------------------------------------------------------------===#

r"""
Clang Static Analyzer test helper
=================================

This script converts a CSV file to a JSON file with a specific structure.

The JSON file contains a single dictionary.  The keys of this dictionary
are taken from the first column of the CSV. The values are dictionaries
themselves, mapping the CSV header names (except the first column) to
the corresponding row values.


Usage:
  csv2json.py <source-file>

Example:
  // RUN: %csv2json.py %t | FileCheck %s
"""

import csv
import sys
import json


def csv_to_json_dict(csv_filepath):
    """
    Args:
        csv_filepath: The path to the input CSV file.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        csv.Error: If there is an error parsing the CSV file.
        Exception: For any other unexpected errors.
    """
    try:
        with open(csv_filepath, "r", encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile, skipinitialspace=True)

            # Read the header row (column names)
            try:
                header = next(reader)
            except StopIteration:  # Handle empty CSV file
                json.dumps({}, indent=2)  # write an empty dict
                return

            # handle a csv file that contains no rows, not even a header row.
            if not header:
                json.dumps({}, indent=2)
                return

            header_length = len(header)
            other_column_names = header[1:]

            data_dict = {}

            for row in reader:
                if len(row) != header_length:
                    raise csv.Error("Inconsistent CSV file")
                    exit(1)

                key = row[0]
                value_map = {}

                for i, col_name in enumerate(other_column_names):
                    # +1 to skip the first column
                    value_map[col_name] = row[i + 1].strip()

                data_dict[key] = value_map

        return json.dumps(data_dict, indent=2)

    except FileNotFoundError:
        raise FileNotFoundError(f"Error: CSV file not found at {csv_filepath}")
    except csv.Error as e:
        raise csv.Error(f"Error parsing CSV file: {e}")
    except Exception as e:
        raise Exception(f"An unexpected error occurred: {e}")


def main():
    """Example usage with error handling."""
    csv_file = sys.argv[1]

    try:
        print(csv_to_json_dict(csv_file))
    except (FileNotFoundError, csv.Error, Exception) as e:
        print(str(e))
    except:
        print("An error occured")


if __name__ == "__main__":
    main()
