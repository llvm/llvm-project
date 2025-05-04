import subprocess
import requests
import yaml
import sys
import difflib
from termcolor import colored

# Function to load configuration from the YAML file
def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)["project"]

# Function to fetch the diff from the GitHub API
def fetch_diff(owner, repo, pr_number):
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}.diff"
    headers = {"Accept": "application/vnd.github.v3.diff"}
    print(f"📥 Fetching diff from {url}")
    resp = requests.get(url, headers=headers)
    if resp.status_code != 200:
        print(f"❌ Failed to fetch diff: {resp.status_code}")
        sys.exit(1)
    return resp.text

# Function to extract only formatted lines (those with +)
def extract_formatted_lines(diff):
    formatted_lines = []
    for line in diff.splitlines():
        if line.startswith('+'):  # Lines added or modified (formatted)
            formatted_lines.append(line[1:])  # Remove the "+" sign
    return formatted_lines

# Function to run clang-tidy on specific files/lines
def run_clang_tidy_on_formatted_lines(files):
    for file in files:
        print(f"🎯 Running clang-tidy on {file}")
        try:
            result = subprocess.run(
                ["clang-tidy", file, "--checks='*'"],  # Run clang-tidy with all checks
                capture_output=True,
                check=True
            )
            print("📤 clang-tidy Output:")
            print(result.stdout.decode())
        except subprocess.CalledProcessError as e:
            print("❌ clang-tidy failed!")
            print("📤 STDOUT:")
            print(e.stdout.decode())
            print("📥 STDERR:")
            print(e.stderr.decode())
            sys.exit(1)

# Main function to load configuration, fetch diff, and run clang-tidy on formatted lines
def main():
    config = load_config()
    diff_text = fetch_diff(config["owner"], config["repo"], config["pr_number"])

    # Extract only the lines that were formatted (added/modified)
    formatted_lines = extract_formatted_lines(diff_text)

    if not formatted_lines:
        print("✅ No formatted lines found.")
    else:
        print("\n🧼 Suggested clang-tidy changes on formatted lines:\n")
        # Display formatted lines to user
        for line in formatted_lines:
            print(colored(line, 'green'))  # Display formatted lines in green

        # Run clang-tidy on the files with formatted lines
        files_to_check = []  # Populate this list with the relevant files
        for line in formatted_lines:
            # Here you can add logic to identify the file and the line number (from the diff)
            file = "<path_to_file>"  # Extract file path from the diff
            files_to_check.append(file)
        
        # Run clang-tidy on the formatted files
        run_clang_tidy_on_formatted_lines(files_to_check)

if __name__ == "__main__":
    main()
