#!/bin/bash

# Check if at least one PR number is provided
if [ $# -lt 1 ]; then
  echo "Usage: $0 <PR1> <PR2> ... <PRn>"
  exit 1
fi

# Loop through each provided PR number
for pr in "$@"; do
  echo "Processing PR #$pr..."

  # Fetch the PR branch
  git fetch origin pull/$pr/head:pr-$pr

  # Create a diff file between the main branch and the PR
  git diff origin/main...pr-$pr > pr-$pr.diff

  # Run clang-tidy-diff.py on the generated diff file
  cat pr-$pr.diff | ./clang-tidy-diff.py -p1 -j4

  # Optionally, you can clean up the diff file after processing
  rm pr-$pr.diff
done

echo "Finished processing all PRs."

