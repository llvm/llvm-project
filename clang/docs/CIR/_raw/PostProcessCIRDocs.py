# This script embeds MLIR generated documentations (in Markdown format) into
# CIRLangRef_Template.md to generate CIRLangRef.md

import os
import sys


docs_src_dir = sys.argv[1]
docs_bin_dir = sys.argv[2]

DIALECT_DOC_PATH = os.path.join(docs_bin_dir, "CIR", "_raw", "CIRDialect.md")
DIALECT_DOC_OUTPUT_PATH = os.path.join(docs_bin_dir, "CIR", "CIRDialect.md")

INDEX_PATH = os.path.join(docs_src_dir, "CIR", "index.rst")
INDEX_OUTPUT_PATH = os.path.join(docs_bin_dir, "CIR", "index.rst")

cir_docs_toctree = []

# ===============================================
# Post-process CIRDialect.md
# ===============================================
if os.path.exists(DIALECT_DOC_PATH):
    cir_docs_toctree.append("CIRDialect")
    with open(DIALECT_DOC_PATH, encoding="utf-8") as fp:
        dialect_doc = fp.read()
    dialect_doc = dialect_doc.replace(
        "[TOC]",
        """
```{contents}
---
local:
depth: 2
---
```""".strip(),
    )
    with open(DIALECT_DOC_OUTPUT_PATH, "w", encoding="utf-8") as fp:
        fp.write(dialect_doc)

# ===============================================
# Add toctree to index.rst if CIR docs are generated
# ===============================================
if len(cir_docs_toctree) > 0:
    with open(INDEX_PATH, encoding="utf-8") as fp:
        index_content = fp.read()
    index_content += """

CIR Dialect Reference
==========================

.. toctree::
    :numbered:
    :maxdepth: 1

    {}
""".format(
        "\n    ".join(cir_docs_toctree)
    )
    with open(INDEX_OUTPUT_PATH, "w", encoding="utf-8") as fp:
        fp.write(index_content)
