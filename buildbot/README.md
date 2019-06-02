# Scripts for build steps on Buildbot

## Purpose

The purpose of the script is for developer to customize build command for the
builder on Buildbot.

## How it works

The scripts will be run by Buildbot at corresponding build step, for example,
the **compile** step will run `compile.py`. Developer can change the build
command, then the builder (e.g. pull request builder) will use the changed
command to do the build.

## Arguments for the scripts

Refer to argument parser in `main()` function.

## Assumptions

The Buildbot worker directory structure is:

    /path/to/WORKER_ROOT/BUILDER/
        llvm.src  --> source code
        llvm.obj  --> build directory

Working directory of the scripts is /path/to/WORKER_ROOT/BUILDER

