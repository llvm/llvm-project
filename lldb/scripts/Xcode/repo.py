import json
import os
import re
import shutil
import subprocess

def identifier():
	try:
		svn_output = subprocess.check_output(["svn", "info", "--show-item", "url"], stderr=subprocess.STDOUT).rstrip()
		return svn_output
	except:
		pass
	try:
		git_remote_and_branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"], stderr=subprocess.STDOUT).rstrip()
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
	override_path = os.path.join(repos_dir, "OVERRIDE")
	if os.path.isfile(override_path):
		override_set = json.load(open(override_path))
		return override_set["repos"]
	fallback_path = os.path.join(repos_dir, "FALLBACK")
	for path in [os.path.join(repos_dir, f) for f in filter(json_regex.match, os.listdir(repos_dir))]:
		fd = open(path)
		set = json.load(fd)
		fd.close()
		if any(re.match(set_regex, identifier) for set_regex in set["regexs"]):
			shutil.copyfile(path, fallback_path)
			return set["repos"]
	if os.path.isfile(fallback_path):
		fallback_set = json.load(open(fallback_path))
		return fallback_set["repos"]
	sys.exit("Couldn't find a branch configuration for " + identifier + " and there was no " + fallback_path)
