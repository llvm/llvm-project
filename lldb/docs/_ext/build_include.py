from docutils.parsers.rst import directives, Directive
from docutils import utils, statemachine

from sphinx.application import Sphinx
import os
from pathlib import Path


class BuildInclude(Directive):
    """
    Directive to include generated files from the build directory (specified by LLDB_BUILD_DIR).
    This is a simplified version of the `include` directive from docutils with the change that paths
    are relative to the build directory.
    """

    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = True
    option_spec = {"parser": directives.parser_name}

    def run(self):
        path = directives.path(self.arguments[0])
        path = utils.relative_path(None, Path(os.environ["LLDB_BUILD_DIR"]) / path)
        with open(path) as f:
            rawtext = f.read()
        include_lines = statemachine.string2lines(
            rawtext, self.state.document.settings.tab_width, convert_whitespace=True
        )

        # parse into a dummy document and return created nodes
        document = utils.new_document(path, self.state.document.settings)
        parser = self.options["parser"]()
        parser.parse("\n".join(include_lines), document)
        # clean up doctree and complete parsing
        document.transformer.populate_from_components((parser,))
        document.transformer.apply_transforms()
        self.state.document.settings.record_dependencies.add(path)
        return document.children


def setup(app: Sphinx):
    app.add_directive("build-include", BuildInclude)

    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
