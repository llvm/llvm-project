"""Sphinx support for Clang attribute syntax tables."""

from __future__ import annotations

import json
from pathlib import Path

from docutils import nodes
from sphinx.errors import ExtensionError
from sphinx.transforms import SphinxTransform


class clang_attr_syntaxes(nodes.General, nodes.Element):
    """Placeholder node for a Clang attribute syntax table."""

    pass


def clang_attr_syntaxes_role(
    name, rawtext, text, lineno, inliner, options=None, content=None
):
    """Create a placeholder node keyed by documentation record name."""
    node = clang_attr_syntaxes(rawtext)
    node["doc"] = text.strip()
    return [node], []


def load_attribute_syntaxes(app):
    """Load the generated Clang attribute syntax manifest."""
    manifest_path = Path(app.srcdir) / app.config.clang_attr_syntaxes_manifest
    try:
        manifest = json.loads(manifest_path.read_text())
    except OSError as e:
        raise ExtensionError(
            f"unable to read Clang attribute syntaxes manifest: {manifest_path}"
        ) from e

    syntaxes = {}
    for entry in manifest["attributes"] + manifest["undocumented"]:
        syntaxes[entry["doc"]] = entry
    return syntaxes


def has_visible_syntax(entry):
    """Return whether an attribute entry should render a syntax table."""
    for syntax in entry["syntaxes"]:
        if syntax.get("supported"):
            return True
        if syntax.get("spellings"):
            return True
    return False


def make_entry(children):
    """Wrap cell content in a table entry paragraph."""
    entry = nodes.entry()
    paragraph = nodes.paragraph()
    paragraph.extend(children)
    entry += paragraph
    return entry


def make_spelling_entry(spellings):
    """Build a table entry containing one literal line per spelling."""
    entry = nodes.entry()
    if not spellings:
        entry += nodes.paragraph()
        return entry

    line_block = nodes.line_block()
    for spelling in spellings:
        line = nodes.line()
        line += nodes.literal(text=spelling)
        line_block += line
    entry += line_block
    return entry


def make_table(entry):
    """Build the rendered syntax table for one attribute entry."""
    table = nodes.table()
    table["classes"].append("clang-attr-syntaxes")

    # Define one equal-width column for each supported syntax family.
    syntaxes = entry["syntaxes"]
    tgroup = nodes.tgroup(cols=len(syntaxes))
    table += tgroup
    for _ in syntaxes:
        tgroup += nodes.colspec(colwidth=1)

    # Render syntax family names as the table header row.
    thead = nodes.thead()
    tgroup += thead
    header_row = nodes.row()
    thead += header_row
    for syntax in syntaxes:
        header_row += make_entry([nodes.Text(syntax["name"])])

    # Render support markers or spelling lists as the single body row.
    tbody = nodes.tbody()
    tgroup += tbody
    body_row = nodes.row()
    tbody += body_row
    for syntax in syntaxes:
        if "supported" in syntax:
            body_row += make_entry([nodes.Text("Yes")] if syntax["supported"] else [])
        else:
            body_row += make_spelling_entry(syntax["spellings"])

    return table


class ClangAttrSyntaxesTransform(SphinxTransform):
    """Replace Clang attribute syntax placeholders with tables."""

    default_priority = 500

    def apply(self):
        """Run the transform over all placeholder nodes in the document."""
        syntaxes = load_attribute_syntaxes(self.app)
        for node in list(self.document.findall(clang_attr_syntaxes)):
            doc = node["doc"]
            if doc not in syntaxes:
                raise ExtensionError(f"unknown Clang attribute syntax section: {doc}")

            parent = node.parent
            replacement = (
                []
                if not has_visible_syntax(syntaxes[doc])
                else make_table(syntaxes[doc])
            )
            if isinstance(parent, nodes.paragraph) and len(parent.children) == 1:
                parent.replace_self(replacement)
            else:
                node.replace_self(replacement)


def setup(app):
    """Register the Clang attribute syntax extension with Sphinx."""
    app.add_config_value(
        "clang_attr_syntaxes_manifest",
        "_generated/AttributeReferenceSyntaxes.json",
        "env",
    )
    app.add_role("clang-attr-syntaxes", clang_attr_syntaxes_role)
    app.add_transform(ClangAttrSyntaxesTransform)
    return {"version": "1.0", "parallel_read_safe": True, "parallel_write_safe": True}
