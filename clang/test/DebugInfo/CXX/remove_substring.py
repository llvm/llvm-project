import os
import sys

def remove_substring_from_files(directory, substring):
    """
    Removes a specified substring from all filenames in the directory.
    Aborts if any duplicates would result.
    """
    if not os.path.isdir(directory):
        print(f"Error: '{directory}' is not a valid directory.")
        sys.exit(1)

    original_to_new = {}
    filenames = os.listdir(directory)

    for filename in filenames:
        new_filename = filename.replace(substring, '')
        if new_filename != filename:
            original_to_new[filename] = new_filename

    # Check for duplicates
    new_names = list(original_to_new.values())
    if len(new_names) != len(set(new_names)):
        print("Error: Removing the substring would result in duplicate filenames. Aborting.")
        for new_name in new_names:
            if new_names.count(new_name) > 1:
                print(f" - Conflict: {new_name}")
        sys.exit(1)

    # Perform renaming
    for old_name, new_name in original_to_new.items():
        old_path = os.path.join(directory, old_name)
        new_path = os.path.join(directory, new_name)
        os.rename(old_path, new_path)
        print(f"Renamed: '{old_name}' → '{new_name}'")

    if not original_to_new:
        print("No files were renamed (substring not found in any filenames).")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Remove a substring from filenames in a directory.")
    parser.add_argument("directory", help="Directory containing files to rename.")
    parser.add_argument("substring", help="Substring to remove from filenames.")

    args = parser.parse_args()
    remove_substring_from_files(args.directory, args.substring)

