#!/bin/bash


# Define the staged files
staged_files=$(git diff --cached --name-only --diff-filter=d -- libcxx/ libcxxabi/ libunwind/)

# Check if any of the staged files have not been properly formatted with git-clang-format

echo "Checking $file..."
formatting_diff=$(git-clang-format --diff -- "$staged_files") # quotes are used here
if [[ "$formatting_diff" != "no modified files to format" && "$formatting_diff" != "clang-format did not modify any files" ]]; then
    echo "$file has not been formatted with git-clang-format."
    git-clang-format --diff -- "$staged_files"
    read -p "Format the files [Y/n]? " -n 1 -r
      echo
    if [[ $REPLY =~ ^[Yy]$ ]]
      then
          git-clang-format -- "$staged_files"
          git add "$staged_files"
          exit 0
    else
          echo "No changes were made to the git config."
          exit 1
    fi

fi


# Everything checks out
echo "All staged files have been formatted with git-clang-format."
exit 0
