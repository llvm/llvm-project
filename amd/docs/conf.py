# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

external_projects_current_project = "llvm-project"

html_theme = "rocm_docs_theme"
html_theme_options = {"flavor": "rocm"}

extensions = ["rocm_docs"]
external_toc_path = "./sphinx/_toc.yml"

version = "22.0.0"
# release = "6.3.2"
html_title = f"llvm-project {version} Documentation"
project = "llvm-project Documentation"
author = "Advanced Micro Devices, Inc."
copyright = (
    "Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved."
)
