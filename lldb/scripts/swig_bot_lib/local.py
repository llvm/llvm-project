#!/usr/bin/env python

"""
Shared functionality used by `client` and `server` when generating or preparing
to generate SWIG on the local machine.
"""

# Future imports
from __future__ import absolute_import
from __future__ import print_function

# Python modules
import argparse
import imp
import io
import logging
import os
import subprocess
import sys
import tempfile
import zipfile

# LLDB modules
import use_lldb_suite

class GenOptions(object):
    src_root = None
    target_dir = None
    languages = None
    swig_executable = None

def pack_archive(config_json, options):
    logging.info("Creating input file package...")
    zip_data = io.BytesIO()
    zip_file = None
    try:
        # It's possible that a custom-built interpreter will not have the
        # standard zlib module.  If so, we can only store, not compress.  By
        # try to compress since we usually have a standard Python distribution.
        zip_file = zipfile.ZipFile(zip_data, mode='w',
                                   compression=zipfile.ZIP_DEFLATED)
    except RuntimeError:
        zip_file = zipfile.ZipFile(zip_data, mode='w',
                                   compression=zipfile.ZIP_STORED)

    filters = [("include/lldb", ".h"),
               ("scripts", ".swig"),
               ("scripts/Python", ".swig"),
               ("scripts/interface", ".i")]
    def filter_func(t):
        subfolder = t[0]
        ext = t[1]
        full_path = os.path.normpath(os.path.join(options.src_root, subfolder))
        candidates = [os.path.normpath(os.path.join(full_path, f))
                      for f in os.listdir(full_path)]
        actual = filter(
            lambda f : os.path.isfile(f) and os.path.splitext(f)[1] == ext,
            candidates)
        return (subfolder, map(lambda f : os.path.basename(f), actual))
    archive_entries = map(filter_func, filters)

    for entry in archive_entries:
        subfolder = entry[0]
        files = entry[1]
        for file in files:
            relative_path = os.path.normpath(os.path.join(subfolder, file))
            full_path = os.path.normpath(
                os.path.join(options.src_root, relative_path))
            logging.info("{} -> {}".format(full_path, relative_path))
            zip_file.write(full_path, relative_path)

    logging.info("(null) -> config.json")
    zip_file.writestr("config.json", config_json)
    zip_file.close()
    return zip_data.getvalue()

def unpack_archive(subfolder, archive_bytes):
    tempfolder = os.path.join(tempfile.gettempdir(), subfolder)
    os.makedirs(tempfolder, exist_ok=True)

    tempfolder = tempfile.mkdtemp(dir=tempfolder)
    logging.debug("Extracting archive to {}".format(tempfolder))

    zip_data = io.BytesIO(archive_bytes)
    logging.debug("Opening zip archive...")
    zip_file = zipfile.ZipFile(zip_data, mode='r')
    zip_file.extractall(tempfolder)
    zip_file.close()
    return tempfolder

def generate(options):
    include_folder = os.path.join(options.src_root, "include")
    in_file = os.path.join(options.src_root, "scripts", "lldb.swig")
    include_folder = os.path.normcase(include_folder)

    for lang in options.languages:
        lang = lang.lower()
        out_dir = os.path.join(options.target_dir, lang.title())
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_file = os.path.join(out_dir, "LLDBWrap{}".format(lang.title()))
        swig_command = [
            options.swig_executable,
            "-c++",
        ]
        swig_command.append("-" + lang)
        if lang == "python":
            swig_command.append("-threads")

        swig_command.extend([
            "-I" + include_folder,
            "-D__STDC_LIMIT_MACROS",
            "-D__STDC_CONSTANT_MACROS",
            "-outdir", out_dir,
            "-o", out_file,
            in_file
        ])

        logging.info("generating swig {} bindings into {}"
                     .format(lang, out_dir))
        logging.debug("swig command line: {}".format(swig_command))
        try:
            # Execute swig
            swig_output = subprocess.check_output(
                swig_command, stderr=subprocess.STDOUT, universal_newlines=True)

            logging.info("swig generation succeeded")
            if swig_output is not None and len(swig_output) > 0:
                logging.info("swig output: %s", swig_output)
        except subprocess.CalledProcessError as e:
            logging.error("An error occurred executing swig.  returncode={}"
                          .format(e.returncode))
            logging.error(e.output)
