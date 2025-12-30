# This program takes in two file path arguments in the form 'compare-file-size.py file1 file2'
# Returns true if the file size of the file1 is smaller than the file size of file2

import argparse
import os


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("file1", type=str)
    parser.add_argument("file2", type=str)

    args = parser.parse_args()

    return os.path.getsize(args.file1) < os.path.getsize(args.file2)


if __name__ == "__main__":
    main()
