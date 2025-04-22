import subprocess
import sys
import re
import requests
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)




PR_NUMBER = str(config["project"]["pr_number"])
OWNER = config["project"]["owner"]
REPO = config["project"]["repo"]

# GitHub API to fetch the PR diff
url = f"https://api.github.com/repos/{OWNER}/{REPO}/pulls/{PR_NUMBER}"
diff_url = f"{url}.diff"

# Fetch the diff from GitHub API
print(f"📥 Fetching PR diff for PR #{PR_NUMBER} from GitHub...")
response = requests.get(diff_url, headers={"Accept": "application/vnd.github.v3.diff"})

if response.status_code != 200:
    print(f"❌ Failed to fetch PR diff or no changes found. Status Code: {response.status_code}")
    sys.exit(1)

# Get the diff content (only .cpp and .h files)
diff_text = response.text
if not diff_text.strip():
    print("✅ No changes in the PR.")
    sys.exit(0)

# Get the list of modified .cpp and .h files in the PR
pr_files = [line.split(" ")[1][2:] for line in diff_text.splitlines() if line.startswith("+++")]
pr_files = [file for file in pr_files if file.endswith(".cpp") or file.endswith(".h")]

if not pr_files:
    print("❌ No relevant .cpp or .h files to check in PR #$PR_NUMBER.")
    sys.exit(0)

# Initialize a list to store missing documentation info
missing_docs = []

# Process each file in the diff
for file in pr_files:
    # Check if the file is a .cpp or .h file
    if file.endswith(".cpp") or file.endswith(".h"):
        # Get the diff for the modified file
        file_diff = "\n".join(
            [line[1:] for line in diff_text.splitlines() if line.startswith(('+', '-')) and line[2:].startswith(file)]
        )

        # Loop through each modified line in the file
        for line in file_diff.splitlines():
            # Check if the line creates a class (i.e., contains "class ")
            if "class " in line:
                # Check the previous line to see if it has Doxygen documentation
                prev_line = None
                lines = file_diff.splitlines()
                idx = lines.index(line)
                if idx > 0:
                    prev_line = lines[idx - 1]

                # If the previous line is not a Doxygen comment, it's missing documentation
                if prev_line and not prev_line.strip().startswith("/**"):
                    missing_docs.append((file, line))
                    print(f"The following class is missing documentation: {file}")
                    print(f"Before: {prev_line}")
                    print(f"After: {line}")
                    print("Action: Please add a Doxygen comment above this class explaining its purpose and functionality.")
                    print("Example:")
                    print("  /**")
                    print("   * @brief Class description: What this class does.")
                    print("   * @details More detailed explanation if needed.")
                    print("   */")
                    print()

# If missing documentation was found, exit with status 1
if missing_docs:
    sys.exit(1)
else:
    print("All modified classes are properly documented.")

print("LLVM CLASS CHECK COMPLETE")
