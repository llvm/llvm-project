# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


import os

from .utils import coalesce_root_path, make_parents, read_json, write_json

BUILDSTEP_RECORD_SUFFIX = ".buildstep.json"


class BuildArtifactId:
    __slots__ = ("abs_path",)

    def __init__(self, build_artifact_path):
        self.abs_path = os.path.realpath(build_artifact_path)

    def get_path(self):
        return self.abs_path

    def __eq__(self, other):
        return isinstance(other, BuildArtifactId) and other.abs_path == self.abs_path

    def __hash__(self):
        return hash(self.abs_path)

    def __repr__(self):
        return "BuildArtifactId(%r)" % (self.abs_path,)


class BuildGraph:
    def __init__(self, build_graph_dir):
        self.build_graph_dir = os.path.abspath(build_graph_dir)

    def get_buildstep_record_path(self, output_build_artifact_id):
        return coalesce_root_path(
            self.build_graph_dir,
            output_build_artifact_id.get_path() + BUILDSTEP_RECORD_SUFFIX,
        )

    def was_built(self, build_artifact_id):
        return os.path.exists(self.get_buildstep_record_path(build_artifact_id))

    def record_build_step(self, output_build_artifact_id, inputs):
        path = make_parents(self.get_buildstep_record_path(output_build_artifact_id))
        write_json(
            path,
            {
                "output": output_build_artifact_id.get_path(),
                "inputs": [each.get_path() for each in inputs],
            },
        )
        return path

    def get_recorded_build_steps(self):
        steps = {}
        for directory, _, filenames in os.walk(self.build_graph_dir):
            for name in sorted(filenames):
                if not name.endswith(BUILDSTEP_RECORD_SUFFIX):
                    continue
                document = read_json(os.path.join(directory, name))
                if document is None or "output" not in document:
                    continue
                steps[document["output"]] = list(document.get("inputs", []))
        return steps

    @staticmethod
    def find_topmost_outputs(steps):
        consumed = set()
        for inputs in steps.values():
            consumed.update(inputs)
        return sorted(output for output in steps if output not in consumed)
