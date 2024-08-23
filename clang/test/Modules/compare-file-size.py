import argparse
import os

def get_file_size(file_path):
    try:
        return os.path.getsize(file_path)
    except:
        print(f"Unable to get file size of {file_path}")
        return None

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("file1", type=str)
    parser.add_argument("file2", type=str)

    args = parser.parse_args()

    return get_file_size(args.file1) < get_file_size(args.file2)

if __name__ == "__main__":
    main()
