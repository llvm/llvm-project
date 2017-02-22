import json
import os
import re
import subprocess

def identifier():
	try:
		svn_output = subprocess.check_output(["svn", "info", "--show-item", "url"], stderr=subprocess.STDOUT).rstrip()
		return svn_output
	except:
		pass
	try:
		git_remote_and_branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"]).rstrip()
		git_remote = git_remote_and_branch.split("/")[0]
		git_branch = "/".join(git_remote_and_branch.split("/")[1:])
		git_url = subprocess.check_output(["git", "remote", "get-url", git_remote]).rstrip()
		return git_url + ":" + git_branch
	except:
		pass
	return None

def find(identifier):
	dir = os.path.dirname(os.path.realpath(__file__))
	repos_dir = os.path.join(dir, "repos")
	json_regex = re.compile(r"^.*.json$")
	override_path = os.path.join(repos_dir, "OVERRIDE.json")
	if os.path.isfile(override_path):
		override_set = json.load(open(override_path))
		return override_set["repos"]
	for set in [json.load(open(os.path.join(repos_dir, f))) for f in filter(json_regex.match, os.listdir(repos_dir))]:
		if re.match(set["regexp"], identifier):
			return set["repos"]
	return None
