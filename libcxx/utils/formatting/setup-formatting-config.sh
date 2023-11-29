#!/bin/bash


# Jump up to the root directory of the repository

git_repo_path="$(git rev-parse --show-toplevel)"
cd "$git_repo_path"

# Set up git-clang-format config so that it formats files without extensions as needed by libc++
function check_formatting_config() {
  # Set the desired git config key and value
  CONFIG_KEY="clangFormat.extensions"
  CONFIG_VALUE=" c,h,m,mm,cpp,cxx,hpp,,"

  # Get the actual value of the config key
  ACTUAL_VALUE=$(git config --local --get $CONFIG_KEY)

  # Check if the actual value is the same as the desired value
  if [[ "$ACTUAL_VALUE" == "$CONFIG_VALUE" ]]; then
      echo "The git config value for $CONFIG_KEY is correctly set to $CONFIG_VALUE"
  else
      echo "Setting up git-clang-format config for libc++..."
      # Prompt the user to set the git config key to the desired value
      echo "Git config key $CONFIG_KEY is not set or incorrect."
      read -p "Would you like to set it to $CONFIG_VALUE [Y/n]? " -n 1 -r
      echo
      if [[ $REPLY =~ ^[Yy]$ ]]
      then
          git config --local $CONFIG_KEY "$CONFIG_VALUE"
          echo "Git config key $CONFIG_KEY has been set to $CONFIG_VALUE"
      else
          echo "No changes were made to the git config."
      fi
  fi
}

# Check for an installation of git-clang-format
function check_clang_format() {
  # Check if git-clang-format is installed
  GIT_CLANG_FORMAT_COMMAND="git-clang-format"
  if command -v $GIT_CLANG_FORMAT_COMMAND >/dev/null 2>&1; then
      echo "git-clang-format is installed in your system."
  else
      echo "Warning: git-clang-format is not installed in your system."
  fi
}
# ...
# Existing script here
# ...

# Check if libcxx-formatting.sh is installed in pre-commit hook
function check_pre_commit_hooks() {
  # Check if libcxx-formatting.sh is present in pre-commit hook
  PRE_COMMIT_FILE=".git/hooks/pre-commit"
  EXPECTED_COMMIT_SCRIPT=". ./libcxx/utils/formatting/libcxx-pre-commit-formatting-hook.sh"
  if grep -q -F "$EXPECTED_COMMIT_SCRIPT" "$PRE_COMMIT_FILE"; then
      echo "libcxx-pre-commit-formatting-hook.sh is already installed in pre-commit hook."
  else
      # Offer to install it
      read -p "libcxx-pre-commit-formatting-hook.sh is not installed. Would you like to install it [Y/n]? " -n 1 -r
      echo
      if [[ $REPLY =~ ^[Yy]$ ]]
      then
          echo "Installing libcxx-pre-commit-formatting-hook.sh..."
          echo "$EXPECTED_COMMIT_SCRIPT" >> "$PRE_COMMIT_FILE"
          echo "Installed libcxx-pre-commit-formatting-hook.sh to pre-commit hook."
      else
          echo "No changes were made to the pre-commit hook."
      fi
  fi
}

check_formatting_config
check_clang_format
check_pre_commit_hooks
