import json
import os

from typing import Any, Dict, List, NamedTuple, Optional


JSON = Dict[str, Any]


DEFAULT_MAP_FILE = "projects.json"


class ProjectInfo(NamedTuple):
    """
    Information about a project to analyze.
    """
    name: str
    mode: int
    enabled: bool = True


class ProjectMap:
    """
    Project map stores info about all the "registered" projects.
    """
    def __init__(self, path: Optional[str] = None, should_exist: bool = True):
        """
        :param path: optional path to a project JSON file, when None defaults
                     to DEFAULT_MAP_FILE.
        :param should_exist: flag to tell if it's an exceptional situation when
                             the project file doesn't exist, creates an empty
                             project list instead if we are not expecting it to
                             exist.
        """
        if path is None:
            path = os.path.join(os.path.abspath(os.curdir), DEFAULT_MAP_FILE)

        if not os.path.exists(path):
            if should_exist:
                raise ValueError(
                    f"Cannot find the project map file {path}"
                    f"\nRunning script for the wrong directory?\n")
            else:
                self._create_empty(path)

        self.path = path
        self._load_projects()

    def save(self):
        """
        Save project map back to its original file.
        """
        self._save(self.projects, self.path)

    def _load_projects(self):
        with open(self.path) as raw_data:
            raw_projects = json.load(raw_data)

            if not isinstance(raw_projects, list):
                raise ValueError(
                    "Project map should be a list of JSON objects")

            self.projects = self._parse(raw_projects)

    @staticmethod
    def _parse(raw_projects: List[JSON]) -> List[ProjectInfo]:
        return [ProjectMap._parse_project(raw_project)
                for raw_project in raw_projects]

    @staticmethod
    def _parse_project(raw_project: JSON) -> ProjectInfo:
        try:
            name: str = raw_project["name"]
            build_mode: int = raw_project["mode"]
            enabled: bool = raw_project.get("enabled", True)
            return ProjectInfo(name, build_mode, enabled)

        except KeyError as e:
            raise ValueError(
                f"Project info is required to have a '{e.args[0]}' field")

    @staticmethod
    def _create_empty(path: str):
        ProjectMap._save([], path)

    @staticmethod
    def _save(projects: List[ProjectInfo], path: str):
        with open(path, "w") as output:
            json.dump(ProjectMap._convert_infos_to_dicts(projects),
                      output, indent=2)

    @staticmethod
    def _convert_infos_to_dicts(projects: List[ProjectInfo]) -> List[JSON]:
        return [project._asdict() for project in projects]
