import json
import os

from enum import Enum
from typing import Any, Dict, List, NamedTuple, Optional, Tuple


JSON = Dict[str, Any]


DEFAULT_MAP_FILE = "projects.json"


class DownloadType(str, Enum):
    GIT = "git"
    ZIP = "zip"
    SCRIPT = "script"


class ProjectInfo(NamedTuple):
    """
    Information about a project to analyze.
    """
    name: str
    mode: int
    source: DownloadType = DownloadType.SCRIPT
    origin: str = ""
    commit: str = ""
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
            source: DownloadType = raw_project.get("source", "zip")

            if source == DownloadType.GIT:
                origin, commit = ProjectMap._get_git_params(raw_project)
            else:
                origin, commit = "", ""

            return ProjectInfo(name, build_mode, source, origin, commit,
                               enabled)

        except KeyError as e:
            raise ValueError(
                f"Project info is required to have a '{e.args[0]}' field")

    @staticmethod
    def _get_git_params(raw_project: JSON) -> Tuple[str, str]:
        try:
            return raw_project["origin"], raw_project["commit"]
        except KeyError as e:
            raise ValueError(
                f"Profect info is required to have a '{e.args[0]}' field "
                f"if it has a 'git' source")

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
        return [ProjectMap._convert_info_to_dict(project)
                for project in projects]

    @staticmethod
    def _convert_info_to_dict(project: ProjectInfo) -> JSON:
        whole_dict = project._asdict()
        defaults = project._field_defaults

        # there is no need in serializing fields with default values
        for field, default_value in defaults.items():
            if whole_dict[field] == default_value:
                del whole_dict[field]

        return whole_dict
