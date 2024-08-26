# This program takes in two file path arguments and returns true if the
# file size of the first file is smaller than the file size of the second file

import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument("file1", type=str)
parser.add_argument("file2", type=str)

args = parser.parse_args()

return os.path.getsize(args.file1) < os.path.getsize(args.file2)
