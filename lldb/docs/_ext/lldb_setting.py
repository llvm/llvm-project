from docutils.parsers.rst import directives
from docutils import nodes

from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.directives import ObjectDescription
from sphinx.util.docfields import Field, GroupedField
import llvm_slug


class LiteralField(Field):
    """A field that wraps the content in <code></code>"""

    def make_field(self, types, domain, item, env=None, inliner=None, location=None):
        fieldarg, content = item
        fieldname = nodes.field_name("", self.label)
        if fieldarg:
            fieldname += nodes.Text(" ")
            fieldname += nodes.Text(fieldarg)

        fieldbody = nodes.field_body("", nodes.literal("", "", *content))
        return nodes.field("", fieldname, fieldbody)


# Example:
# ```{lldbsetting} dwim-print-verbosity
# :type: "enum"
#
# The verbosity level used by dwim-print.
#
# :enum none: Use no verbosity when running dwim-print.
# :enum expression: Use partial verbosity when running dwim-print - display a message when `expression` evaluation is used.
# :enum full: Use full verbosity when running dwim-print.
# :default: none
# ```
class LLDBSetting(ObjectDescription):
    option_spec = {
        "type": directives.unchanged,
    }
    doc_field_types = [
        LiteralField(
            "default",
            label="Default",
            has_arg=False,
            names=("default",),
        ),
        GroupedField("enum", label="Enumerations", names=("enum",)),
    ]

    def handle_signature(self, sig: str, signode: addnodes.desc_signature):
        typ = self.options.get("type", None)

        desc = addnodes.desc_name(text=sig)
        desc += nodes.inline(
            "",
            typ,
            classes=[
                "lldb-setting-type",
                f"lldb-setting-type-{llvm_slug.make_slug(typ)}",
            ],
        )
        signode["ids"].append(sig)
        signode += desc


def setup(app: Sphinx):
    app.add_directive("lldbsetting", LLDBSetting)

    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
