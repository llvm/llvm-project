"""
Module Name: build_metadata_extractor.py

Description:
    This module facilitates the extraction of build metadata from nightly
    builds. It reads a manifest file, collects data about submodules,
    generates tables for artifacts and build logs, and lists any failed
    jobs from GitHub actions. The results are saved in a structured JSON
    file for further analysis and reporting.

Key Classes and Functions:
    - BuildMetadataExtractor: Main class containing methods to extract and
      compile metadata.
        - __init__(...): Initializes with required parameters for accessing
          GitHub and reading manifests.
        - read_manifest_file(): Reads and validates the manifest JSON file.
        - extract_submodule_table(manifest_data): Generates a table of
          submodules and their URLs.
        - generate_manifest_artifact_logs_table(): Creates a table for Rock
          Manifest, Artifacts, and Build Logs.
        - list_failures(): Fetches and lists all failure jobs, handling
          potential pagination.
        - save_results_to_file(...): Saves the results to a file in
          structured JSON format.

Environment Variables:
    - ORG_NAME: Organization name on GitHub.
    - PROJECT_NAME: Project name within the organization.
    - RUN_ID: Identifier for the GitHub Actions workflow run.
    - GITHUB_TOKEN: GitHub API token for authentication.
    - MANIFEST_FILE: Path to the manifest file to be read.
    - ROCK_MANIFEST_URL: URL for the Rock manifest.
    - ARTIFACTS_URL: URL for the build artifacts.
    - BUILD_LOGS_URL: URL for accessing build logs.
    - OUTPUT_FILE: File name for saving extracted metadata results.
"""

import json
import os
import requests


class BuildMetadataExtractor:
    def __init__(self, org_name, project_name, run_id, github_token, manifest_path, rock_manifest_url, artifacts_url, build_logs_url, output_file):
        """Initialize the extractor with organization, project info, and additional inputs."""
        self.org_name = org_name
        self.project_name = project_name
        self.run_id = run_id
        self.github_token = github_token
        self.manifest_path = manifest_path
        self.rock_manifest_url = rock_manifest_url
        self.artifacts_url = artifacts_url
        self.build_logs_url = build_logs_url
        self.output_file = output_file
        self.headers = {'Authorization': f'token {self.github_token}'}
        self.base_url = f"https://api.github.com/repos/{self.org_name}/{self.project_name}"

    def read_manifest_file(self):
        """Reads and validates the manifest JSON file."""
        if not os.path.exists(self.manifest_path):
            print(f"Manifest file '{self.manifest_path}' does not exist.")
            return None
        with open(self.manifest_path, 'r') as f:
            return json.load(f)

    def extract_submodule_table(self, manifest_data):
        """Generates a table detailing submodules and URLs."""
        the_rock_commit = manifest_data.get('the_rock_commit', '')
        commit_url = f"{self.base_url}/commit/{the_rock_commit}"
        table = '| Submodule | URL |\n|-----------|-----|\n'

        for submodule in manifest_data.get('submodules', []):
            submodule_name = submodule['submodule_name']
            if submodule_name == 'rccl':  # Filter out specific submodules
                continue
            pin_sha = submodule.get('pin_sha')
            if pin_sha:
                submodule_url = submodule['submodule_url']
                commit_url = f"{submodule_url.replace('.git', '')}/commit/{pin_sha}"
                table += f'| {submodule_name} | ({commit_url}) |\\n'

        return table

    def generate_manifest_artifact_logs_table(self):
        """Creates a table for Rock Manifest, Artifacts, and Build Logs."""
        table = '| Description | URL |\n|-------------|-----|\n'
        table += f'| Rock Manifest | ({self.rock_manifest_url}) |\\n'
        table += f'| Artifacts | ({self.artifacts_url}) |\\n'
        table += f'| Build Logs | ({self.build_logs_url}) |\\n'

        return table

    def list_failures(self):
        """Fetches and lists all failure jobs, handling potential pagination."""
        jobs_url = f"{self.base_url}/actions/runs/{self.run_id}/jobs"
        failure_jobs = []
        page = 1

        while True:
            response = requests.get(f"{jobs_url}?page={page}", headers=self.headers)
            jobs_data = response.json()
            if 'jobs' not in jobs_data or not jobs_data['jobs']:
                break

            failure_jobs.extend([job for job in jobs_data['jobs'] if job.get('conclusion') == 'failure'])
            page += 1

        if not failure_jobs:
            return None  # No failures, return None

        failure_table = '| Failure Job Name | Job URL |\n|------------------|---------|\n'
        for job in failure_jobs:
            job_name = job['name']
            job_url = job['html_url']
            failure_table += f'| {job_name} | ({job_url}) |\\n'
        return failure_table

    def save_results_to_file(self, submodule_table, manifest_artifacts_table, failure_table):
        """Saves the results to a file in structured JSON format."""
        results = {
            "submodule_table": submodule_table,
            "manifest_artifacts_table": manifest_artifacts_table,
            "failure_table": failure_table or "No failures found"
        }
        with open(self.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results have been saved to {self.output_file}")

if __name__ == "__main__":
    # Initialize variables from environment
    ORG_NAME = os.getenv("ORG_NAME", "ROCm")
    PROJECT_NAME = os.getenv("PROJECT_NAME", "llvm-project")
    RUN_ID = os.getenv("RUN_ID", "")
    GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
    MANIFEST_FILE = os.getenv("MANIFEST_FILE", "manifest.json")
    ROCK_MANIFEST_URL = os.getenv("ROCK_MANIFEST_URL")
    ARTIFACTS_URL = os.getenv("ARTIFACTS_URL")
    BUILD_LOGS_URL = os.getenv("BUILD_LOGS_URL")
    OUTPUT_FILE = os.getenv("OUTPUT_FILE", "results.json")

    # Initialize extractor
    extractor = BuildMetadataExtractor(
        ORG_NAME, PROJECT_NAME, RUN_ID, GITHUB_TOKEN, MANIFEST_FILE, 
        ROCK_MANIFEST_URL, ARTIFACTS_URL, BUILD_LOGS_URL, OUTPUT_FILE
    )

    # Process the manifest file
    manifest_data = extractor.read_manifest_file()
    if manifest_data:
        submodule_table = extractor.extract_submodule_table(manifest_data)
        manifest_artifacts_table = extractor.generate_manifest_artifact_logs_table()
        failure_table = extractor.list_failures()

        # Save results to an output file
        extractor.save_results_to_file(submodule_table, manifest_artifacts_table, failure_table)

