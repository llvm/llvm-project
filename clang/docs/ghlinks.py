#!/usr/bin/env python3

"""Script to link #GH123 to our github issue tracker.

We use MyST and reST, so this Sphinx plugin operates on the shared doctree
representation of the documentation, which is effectively a documentation-AST.

Docutils doctree node reference:
https://docutils.sourceforge.io/docs/ref/doctree.html
"""

import argparse
import re
import sys
import tempfile
import textwrap
from pathlib import Path


def _find_docs_requirements() -> str:
    for parent in Path(__file__).resolve().parents:
        docs_requirements = parent / "llvm/docs/requirements.txt"
        if docs_requirements.exists():
            return str(docs_requirements)
    return "llvm/docs/requirements.txt"


try:
    from docutils import nodes
    from sphinx.application import Sphinx
except ImportError as err:
    print(
        textwrap.dedent(
            f"""
            ghlinks.py requires the LLVM documentation build dependencies.

            Import failed with:
              {err}

            The standard requirements file is:
              {_find_docs_requirements()}

            From an llvm-project checkout, a typical pip setup is:
              python3 -m venv .venv
              . .venv/bin/activate
              python3 -m pip install -r llvm/docs/requirements.txt
              python3 clang/docs/ghlinks.py --test

            With uv, a typical one-shot command is:
              uv run --with-requirements llvm/docs/requirements.txt \\
                python clang/docs/ghlinks.py --test
            """
        ).strip(),
        file=sys.stderr,
    )
    raise

__version__ = "1.0"

GH_LINK_RE = re.compile("#GH([1-9][0-9]+)")
GH_LINK_TMPL = "https://github.com/llvm/llvm-project/issues/{}"
SKIP_NODES: tuple[type[nodes.Node], ...] = (
    nodes.FixedTextElement,
    nodes.literal,
    nodes.raw,
    nodes.reference,
)


def make_gh_link(issue: str) -> nodes.reference:
    """Create the docutils node that writers render as an external link."""
    return nodes.reference("", "#" + issue, refuri=GH_LINK_TMPL.format(issue))


def replace_gh_links(node: nodes.Text) -> None:
    """Replace one text node with text fragments and GitHub issue links.

    docutils text nodes cannot contain child nodes, so a single string like
    "See #GH123." has to become a sibling list: Text("See "), reference(...),
    Text("."). The parent node owns that sibling list.
    """
    remaining = str(node)
    replacements = []

    while GH_LINK_RE.search(remaining):
        before, issue, remaining = GH_LINK_RE.split(remaining, maxsplit=1)
        if before:
            replacements.append(nodes.Text(before))
        replacements.append(make_gh_link(issue))

    # If we found matches, do the replacement.
    if replacements:
        if remaining:
            replacements.append(nodes.Text(remaining))
        node.parent.replace(node, replacements)


def replace_gh_links_in_subtree(node: nodes.Node) -> None:
    """Rewrite linkable text nodes under node, pruning ignored subtrees.

    The doctree is the markup-language neutral documentation AST, so it
    handles both reST and MyST (markdown). It helps us avoid rewriting #GH123
    in code and literal blocks.
    """
    # DFS cutoff for code blocks etc.
    if isinstance(node, SKIP_NODES):
        return

    # Find #GH123 links in text blocks and linkify them.
    if isinstance(node, nodes.Text):
        if GH_LINK_RE.search(str(node)):
            replace_gh_links(node)
        return

    # Recursive DFS traversal of children.
    if isinstance(node, nodes.Element):
        for child in list(node.children):
            replace_gh_links_in_subtree(child)


def subst_gh_links(_app: Sphinx, doctree: nodes.document) -> None:
    """Link #GH123 references after the source has been parsed."""
    replace_gh_links_in_subtree(doctree)


def setup(app: Sphinx) -> dict[str, object]:
    app.connect("doctree-read", subst_gh_links)
    return dict(version=__version__, parallel_read_safe=True, parallel_write_safe=True)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="store_true", help="run ghlinks self-tests")
    args = parser.parse_args(argv)

    if args.test:
        run_tests()
        print(
            "ghlinks.py: tests passed; next, rebuild docs-clang-html and spot check the release notes"
        )
        return 0

    parser.print_help(sys.stderr)
    return 0


# -----------------------------------------------------------------------------
# Test code only below:
# -----------------------------------------------------------------------------

REST_TEST_DOC = r"""
GHLink reST Test
================

A paragraph links #GH123 and #GH456.

No leading zero link #GH0123.

Inline literal ``#GH333`` stays text.

Existing link `#GH666 <https://example.com/rst>`_ stays existing.

Code block::

  #GH777

.. raw:: html

   <span>#GH888</span>
"""


MARKDOWN_TEST_DOC = r"""
# GHLink Markdown Test

A paragraph links #GH234 and #GH567.

No leading zero link #GH0234.

Inline code `#GH444` stays text.

Existing link [#GH999](https://example.com/md) stays existing.

```c
#GH778
```
"""


def _build_test_docs() -> tuple[str, str]:
    """Build in-file reST and Markdown test strings and return their HTML."""
    conf = f"""
import sys
sys.path.insert(0, {str(Path(__file__).parent)!r})

extensions = ["ghlinks", "myst_parser"]
master_doc = "index"
project = "ghlinks test"
source_suffix = {{
    ".rst": "restructuredtext",
    ".md": "markdown",
}}
"""
    index = """
GHLink Tests
============

.. toctree::

   rest
   markdown
"""

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        srcdir = tmp_path / "src"
        outdir = tmp_path / "out"
        doctreedir = tmp_path / "doctrees"
        srcdir.mkdir()
        (srcdir / "conf.py").write_text(textwrap.dedent(conf), encoding="utf-8")
        (srcdir / "index.rst").write_text(textwrap.dedent(index), encoding="utf-8")
        (srcdir / "rest.rst").write_text(
            textwrap.dedent(REST_TEST_DOC), encoding="utf-8"
        )
        (srcdir / "markdown.md").write_text(
            textwrap.dedent(MARKDOWN_TEST_DOC), encoding="utf-8"
        )

        app = Sphinx(
            srcdir=srcdir,
            confdir=srcdir,
            outdir=outdir,
            doctreedir=doctreedir,
            buildername="html",
            freshenv=True,
            warningiserror=True,
            status=None,
            warning=None,
        )
        app.build()
        return (
            (outdir / "rest.html").read_text(encoding="utf-8"),
            (outdir / "markdown.html").read_text(encoding="utf-8"),
        )


def _issue_href(issue: str) -> str:
    return f'href="{GH_LINK_TMPL.format(issue)}"'


def _check_contains(html: str, needle: str) -> None:
    if needle not in html:
        raise AssertionError(f"expected HTML to contain: {needle}")


def run_tests() -> None:
    rest_html, markdown_html = _build_test_docs()

    for issue in ("123", "456"):
        _check_contains(rest_html, _issue_href(issue))
    for issue in ("0123", "333", "666", "777", "888"):
        _check_contains(rest_html, f"#GH{issue}")
    _check_contains(rest_html, 'href="https://example.com/rst"')

    for issue in ("234", "567"):
        _check_contains(markdown_html, _issue_href(issue))
    for issue in ("0234", "444", "999", "778"):
        _check_contains(markdown_html, f"#GH{issue}")
    _check_contains(markdown_html, 'href="https://example.com/md"')


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
